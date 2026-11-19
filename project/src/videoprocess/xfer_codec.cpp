#include "src/videoprocess/xfer_codec.h"

#include <QDebug>
#include <algorithm>

#include "nvcomp/cascaded.h"   // 3rdparty/nvcomp
#include "nvcomp/native/bitcomp.h"
#include "nvcomp/ans.h"

// cuda_kernels/xfer_compress.cu
extern "C" cudaError_t xfer_fwd(const uint8_t* src, size_t srcPitch, uint8_t* dst, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st);
extern "C" cudaError_t xfer_inv(const uint8_t* src, uint8_t* dst, size_t dstPitch, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st);
extern "C" cudaError_t xfer_pack(const void* const* slots, const size_t* compBytes, int n,
                                 uint64_t* devHdr, uint64_t* hostHdr, uint8_t* hostData, cudaStream_t st);
extern "C" cudaError_t xfer_split(size_t* bytes, int n, size_t chunk, cudaStream_t st);
extern "C" cudaError_t xfer_fetch(const uint64_t* hostHdr, uint64_t* devHdr, int n, const uint8_t* hostData,
                                  uint8_t* devData, size_t copyBytes, size_t maxChunk, size_t cap,
                                  const void** compPtrs, size_t* compBytes, int* flag, cudaStream_t st);
extern "C" cudaError_t xfer_status(const int* status, int n, const size_t* outBytes, const size_t* expBytes,
                                   int* flag, cudaStream_t st);

namespace {
size_t align16(size_t v) { return (v + 15) & ~size_t(15); }

// 時間を測るイベント（DisableTiming なし）を n 個
bool createEvents(std::vector<cudaEvent_t>& v, int n)
{
    v.assign(n, nullptr);
    bool ok = true;
    for (cudaEvent_t& e : v) ok = cudaEventCreate(&e) == cudaSuccess && ok;
    return ok;
}
void destroyEvents(std::vector<cudaEvent_t>& v)
{
    for (cudaEvent_t e : v) if (e) cudaEventDestroy(e);
    v.clear();
}

// nvCOMP のオプション（データ型: NV12 = 1 バイト、P010 = 2 バイト。ANS はバイト列のみ）
nvcompBatchedCascadedCompressOpts_t cascadedOpts(int bps)
{
    nvcompBatchedCascadedCompressOpts_t o = nvcompBatchedCascadedCompressDefaultOpts;
    o.common_opts.data_type = (bps == 2) ? NVCOMP_TYPE_USHORT : NVCOMP_TYPE_UCHAR;
    return o;
}
nvcompBatchedCascadedDecompressOpts_t cascadedDOpts(int bps)
{
    nvcompBatchedCascadedDecompressOpts_t o = nvcompBatchedCascadedDecompressDefaultOpts;
    o.common_opts = cascadedOpts(bps).common_opts;
    return o;
}
nvcompBatchedANSCompressOpts_t ansOpts()
{
    nvcompBatchedANSCompressOpts_t o = nvcompBatchedANSCompressDefaultOpts;
    o.data_type = NVCOMP_TYPE_UCHAR;
    return o;
}
} // namespace

XferCodec::Algo XferCodec::parse(const QString& s)
{
    if (s == "cascaded") return Cascaded;
    if (s == "bitcomp")  return Bitcomp;
    if (s == "ans")      return ANS;
    return None;
}

QString XferCodec::name(Algo a)
{
    switch (a) {
    case Cascaded: return "Cascaded";
    case Bitcomp:  return "Bitcomp";
    case ANS:      return "ANS";
    default:       return "なし";
    }
}

bool XferCodec::init(int srcGpu, int dstGpu, Algo algo, bool delta, int bps,
                     const std::vector<int>& tileW, const std::vector<int>& tileH, int nStage, int ringSize)
{
    release();
    src_ = srcGpu; dst_ = dstGpu; algo_ = algo; delta_ = delta; bps_ = bps; nStage_ = nStage; ringSize_ = ringSize;
    if (algo == None || tileW.empty() || tileW.size() != tileH.size()) return false;

    // 詰めた並び: タイルごとに Y（w × h、差は 1 つ左）、UV（uw × h/2、U と V が交互なので差は 2 つ左）
    //   uw = w を偶数に切り上げ（幅が奇数でも最後の U / V の組を落とさない。例: 1366 の 2 分割 = 683）
    planes_.clear();
    size_t off = 0;
    for (size_t t = 0; t < tileW.size(); t++) {
        const int w = tileW[t], h = tileH[t], uw = (w + 1) & ~1;
        planes_.push_back({ off, w, h, 1 });
        off = align16(off + size_t(w) * h * bps);
        planes_.push_back({ off, uw, h / 2, 2 });
        off = align16(off + size_t(uw) * (h / 2) * bps);
    }
    total_ = off;
    n_ = int((total_ + chunk_ - 1) / chunk_);

    // 1 チャンクの圧縮後の最大、作業領域
    nvcompStatus_t st = nvcompSuccess;
    size_t maxc = 0, ctemp = 0, dtemp = 0;
    switch (algo_) {
    case Cascaded:
        st = nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_, cascadedOpts(bps_), &maxc);
        if (st == nvcompSuccess) st = nvcompBatchedCascadedCompressGetTempSize(n_, chunk_, cascadedOpts(bps_), &ctemp, total_, 0);
        if (st == nvcompSuccess) st = nvcompBatchedCascadedDecompressGetTempSize(n_, chunk_, cascadedDOpts(bps_), &dtemp, total_, 0);
        break;
    case Bitcomp:
        // ネイティブ API（バッチ API は呼び出しで CPU が止まったため）。全体を 1 本で圧縮し、
        //   メインメモリの受け渡しは圧縮後を 64KB の仮想チャンクに切って pack / fetch をそのまま使う
        maxc = chunk_;
        n_ = int((bitcompMaxBuflen(total_) + chunk_ - 1) / chunk_);
        break;
    case ANS:
        st = nvcompBatchedANSCompressGetMaxOutputChunkSize(chunk_, ansOpts(), &maxc);
        if (st == nvcompSuccess) st = nvcompBatchedANSCompressGetTempSize(n_, chunk_, ansOpts(), &ctemp, total_, 0);
        if (st == nvcompSuccess) st = nvcompBatchedANSDecompressGetTempSize(n_, chunk_, nvcompBatchedANSDecompressDefaultOpts, &dtemp, total_, 0);
        break;
    default: break;
    }
    if (st != nvcompSuccess) { qWarning() << "[xfer] nvCOMP の初期化に失敗:" << int(st); return false; }
    maxComp_ = align16(maxc);

    // チャンクの位置と大きさ（両方の GPU で同じ並び）
    std::vector<size_t> chunkBytes(n_);
    for (int i = 0; i < n_; i++)
        chunkBytes[i] = size_t(i) * chunk_ < total_ ? std::min(chunk_, total_ - size_t(i) * chunk_) : 0;

    bool ok = true;
    auto chk = [&](cudaError_t e) { if (e != cudaSuccess) ok = false; };
    const bitcompDataType_t bcType = (bps_ == 2) ? BITCOMP_UNSIGNED_16BIT : BITCOMP_UNSIGNED_8BIT;
    auto makePlan = [&](void** plan, cudaStream_t stream) {
        bitcompHandle_t h = nullptr;
        if (bitcompCreatePlan(&h, total_, bcType, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_ALGO) != BITCOMP_SUCCESS) { ok = false; return; }
        *plan = h;
        if (stream && bitcompSetStream(h, stream) != BITCOMP_SUCCESS) ok = false;
    };

    // ---- 送る側の GPU（src_。デコード: 別 GPU、エンコード: 表示用 GPU）----
    cudaSetDevice(src_);
    chk(cudaMalloc(&sInBytes_, sizeof(size_t) * n_));
    if (ok) chk(cudaMemcpy(sInBytes_, chunkBytes.data(), sizeof(size_t) * n_, cudaMemcpyHostToDevice));
    sTempBytes_ = ctemp;
    nextSet_ = 0;
    for (SrcSet& k : sets_) {
        k.used = false;
        chk(cudaStreamCreateWithFlags(&k.stream, cudaStreamNonBlocking));
        if (algo_ == Bitcomp) makePlan(&k.bcPlan, k.stream);
        chk(cudaEventCreateWithFlags(&k.ready, cudaEventDisableTiming));
        chk(cudaEventCreateWithFlags(&k.done, cudaEventDisableTiming));
        chk(cudaMalloc(&k.tight, total_));
        chk(cudaMalloc(&k.inPtrs, sizeof(void*) * n_));
        chk(cudaMalloc(&k.slotBuf, maxComp_ * n_));
        chk(cudaMalloc(&k.slotPtrs, sizeof(void*) * n_));
        chk(cudaMalloc(&k.compBytes, sizeof(size_t) * n_));
        chk(cudaMalloc(&k.statuses, sizeof(nvcompStatus_t) * n_));
        if (ok) chk(cudaMemset(k.statuses, 0, sizeof(nvcompStatus_t) * n_));   // 0 = 成功（版によって圧縮側が書かなくても誤検出しない）
        chk(cudaMalloc(&k.hdr, sizeof(uint64_t) * (2 + 2 * size_t(n_))));
        if (ctemp) chk(cudaMalloc(&k.temp, ctemp));
        if (!ok) break;
        std::vector<void*> in(n_), slot(n_);
        for (int i = 0; i < n_; i++) { in[i] = k.tight + size_t(i) * chunk_; slot[i] = k.slotBuf + size_t(i) * maxComp_; }
        chk(cudaMemcpy(k.inPtrs, in.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
        chk(cudaMemcpy(k.slotPtrs, slot.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
    }
    used_.assign(nStage_, false);
    ok = createEvents(encBeg_, nStage_) && createEvents(cmpEnd_, nStage_)
      && createEvents(sendBeg_, nStage_) && createEvents(encEnd_, nStage_) && ok;

    // ---- メインメモリ（Portable: どの GPU からも使える、Mapped: カーネルから直接書ける）----
    //   hFlag_: GPU 側で見つけた失敗（[0] 圧縮、[1] 展開。nvCOMP の状態・ヘッダの検査）。CPU は deviceError で読む
    chk(cudaHostAlloc((void**)&hFlag_, sizeof(int) * 2, cudaHostAllocPortable | cudaHostAllocMapped));
    if (hFlag_) { hFlag_[0] = 0; hFlag_[1] = 0; }
    hHdr_.assign(nStage_, nullptr); hData_.assign(nStage_, nullptr);
    for (int s = 0; s < nStage_; s++) {
        chk(cudaHostAlloc((void**)&hHdr_[s], sizeof(uint64_t) * (2 + 2 * size_t(n_)), cudaHostAllocPortable | cudaHostAllocMapped));
        chk(cudaHostAlloc((void**)&hData_[s], maxComp_ * n_, cudaHostAllocPortable | cudaHostAllocMapped));
    }

    // ---- 受ける側の GPU（dst_。デコード: 表示用 GPU、エンコード: 別 GPU）----
    cudaSetDevice(dst_);
    chk(cudaMalloc(&dData_, maxComp_ * n_));
    chk(cudaMalloc(&dHdr_, sizeof(uint64_t) * (2 + 2 * size_t(n_))));
    if (algo_ == Bitcomp) makePlan(&dPlan_, nullptr);   // ストリームは decompress で結び付ける
    lastComp_ = 0;
    chk(cudaMalloc(&dCompPtrs_, sizeof(void*) * n_));
    chk(cudaMalloc(&dCompBytes_, sizeof(size_t) * n_));
    // 展開先: 普段は 1 面（差分を戻して転送先へ書く）。詰めたまま渡すときはリングの段数ぶん（GUI が読む間は残す）
    const int nOut = ringSize_ > 0 ? ringSize_ : 1;
    dTight_.assign(nOut, nullptr);
    dOutPtrs_.assign(nOut, nullptr);
    for (int j = 0; j < nOut; j++) {
        chk(cudaMalloc(&dTight_[j], total_));
        chk(cudaMalloc(&dOutPtrs_[j], sizeof(void*) * n_));
    }
    chk(cudaMalloc(&dOutBufBytes_, sizeof(size_t) * n_));
    chk(cudaMalloc(&dOutBytes_, sizeof(size_t) * n_));
    chk(cudaMalloc(&dStatus_, sizeof(nvcompStatus_t) * n_));
    if (ok) chk(cudaMemset(dStatus_, 0, sizeof(nvcompStatus_t) * n_));
    if (dtemp) { chk(cudaMalloc(&dTemp_, dtemp)); dTempBytes_ = dtemp; }
    if (ok) {
        std::vector<void*> out(n_);
        for (int j = 0; j < nOut && ok; j++) {
            for (int i = 0; i < n_; i++) out[i] = dTight_[j] + size_t(i) * chunk_;
            chk(cudaMemcpy(dOutPtrs_[j], out.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
        }
        chk(cudaMemcpy(dOutBufBytes_, chunkBytes.data(), sizeof(size_t) * n_, cudaMemcpyHostToDevice));
        // 展開後の大きさの初期値も元の大きさに（版によって nvCOMP が書かなくても誤検出しない。xfer_status）
        chk(cudaMemcpy(dOutBytes_, chunkBytes.data(), sizeof(size_t) * n_, cudaMemcpyHostToDevice));
    }
    ok = createEvents(decBeg_, nStage_) && createEvents(fetchEnd_, nStage_)
      && createEvents(decompEnd_, nStage_) && createEvents(decEnd_, nStage_) && ok;

    if (!ok) {
        qWarning() << "[xfer] メモリの確保に失敗:" << cudaGetErrorString(cudaGetLastError());
        release();
        cudaSetDevice(dst_);
        return false;
    }
    ready_ = true;
    qDebug().noquote() << QString("[xfer] GPU%1 → GPU%2 を %3 で圧縮転送（%4 チャンク × %5 KB、1 フレーム %6 MB%7）")
                              .arg(src_).arg(dst_).arg(name(algo_)).arg(n_).arg(chunk_ / 1024)
                              .arg(total_ / 1e6, 0, 'f', 1).arg(delta_ ? "、差分あり" : "");
    return true;
}

// デコード用の入口（cuvid の AVFrame）。前処理（xfer_fwd）をして、既定ストリームの順序で圧縮
bool XferCodec::compress(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree)
{
    if (!ready_ || frames.size() * 2 != planes_.size()) return false;
    std::vector<Src> src;
    src.reserve(frames.size());
    for (const AVFrame* f : frames)
        src.push_back({ f->data[0], size_t(f->linesize[0]), f->data[1], size_t(f->linesize[1]) });
    return compressImpl(s, &src, nullptr, stageFree);
}

// エンコード用（変換カーネルが詰めた並びを書き済み。nextTight と組）
uint8_t* XferCodec::nextTight(cudaEvent_t* reuseDone) const
{
    const SrcSet& k = sets_[nextSet_];
    if (reuseDone) *reuseDone = k.used ? k.done : nullptr;
    return k.tight;
}

bool XferCodec::compressPacked(int s, cudaEvent_t inputReady, cudaEvent_t stageFree)
{
    if (!ready_ || !inputReady) return false;
    return compressImpl(s, nullptr, inputReady, stageFree);
}

// 圧縮の本体（デコード・エンコード共通）
//   src あり（デコード）: xfer_fwd で前処理。inputReady = nullptr なので既定ストリームの後から
//   src = nullptr（エンコード）: 詰めた並び（組の tight）は変換カーネルが書き済み。inputReady の後から
bool XferCodec::compressImpl(int s, const std::vector<Src>* src, cudaEvent_t inputReady, cudaEvent_t stageFree)
{
    // CUDA API の失敗は ok に集める（1 つでも失敗したらこのフレームは失敗。呼び出し側が扱う）
    bool ok = true;
    auto ck = [&](cudaError_t e) { if (e != cudaSuccess) ok = false; };
    ck(cudaSetDevice(src_));
    SrcSet& k = sets_[nextSet_];
    nextSet_ = (nextSet_ + 1) % kSets;
    if (inputReady) {
        // 変換カーネルの完了の後から（エンコード。組の詰めた並びの再利用は呼び出し側が nextTight の reuseDone で守る）
        ck(cudaStreamWaitEvent(k.stream, inputReady, 0));
    } else {
        // 以降の cuvid の出力コピーが、この組の前回のフレームの面を読み終える前に上書きしないための保険
        if (k.used) ck(cudaStreamWaitEvent(0, k.done, 0));
        // 既定ストリーム（cuvid の出力コピー）の後から、組のストリームで進める
        ck(cudaEventRecord(k.ready, 0));
        ck(cudaStreamWaitEvent(k.stream, k.ready, 0));
    }
    cudaStream_t cs = k.stream;
    ck(cudaEventRecord(encBeg_[s], cs));
    for (size_t t = 0; src && t < src->size(); t++) {
        const Src& f = (*src)[t];
        const Plane& py = planes_[2 * t];
        const Plane& pu = planes_[2 * t + 1];
        ok = ok && xfer_fwd(f.y,  f.yPitch,  k.tight + py.off, py.elems, py.rows, bps_, py.lag, delta_, cs) == cudaSuccess
                && xfer_fwd(f.uv, f.uvPitch, k.tight + pu.off, pu.elems, pu.rows, bps_, pu.lag, delta_, cs) == cudaSuccess;
    }
    nvcompStatus_t st = nvcompSuccess;
    switch (algo_) {
    case Cascaded:
        st = nvcompBatchedCascadedCompressAsync(k.inPtrs, sInBytes_, chunk_, n_, k.temp, sTempBytes_,
                                               k.slotPtrs, k.compBytes, cascadedOpts(bps_), (nvcompStatus_t*)k.statuses, cs);
        break;
    case Bitcomp: {
        // 圧縮後の大きさを compBytes[0] に受けて、仮想チャンクの大きさに切り分ける
        const bool b = bitcompCompressLossless(static_cast<bitcompHandle_t>(k.bcPlan), k.tight, k.slotBuf) == BITCOMP_SUCCESS
                    && bitcompGetCompressedSizeAsync(k.slotBuf, k.compBytes, cs) == BITCOMP_SUCCESS
                    && xfer_split(k.compBytes, n_, chunk_, cs) == cudaSuccess;
        st = b ? nvcompSuccess : nvcompErrorInternal;
        break;
    }
    case ANS:
        st = nvcompBatchedANSCompressAsync(k.inPtrs, sInBytes_, chunk_, n_, k.temp, sTempBytes_,
                                          k.slotPtrs, k.compBytes, ansOpts(), (nvcompStatus_t*)k.statuses, cs);
        break;
    default: st = nvcompErrorInvalidValue; break;
    }
    // nvCOMP の戻り値は起動の成否だけ。チャンクごとの結果（statuses）は GPU 上で検査して hFlag_[0] へ（Bitcomp は状態を返さない）
    if (st == nvcompSuccess && algo_ != Bitcomp)
        ck(xfer_status(k.statuses, n_, nullptr, nullptr, &hFlag_[0], cs));
    ck(cudaEventRecord(cmpEnd_[s], cs));
    // メインメモリの枠 s を受ける側の GPU が読み終えるまで待ってから書く
    if (stageFree) ck(cudaStreamWaitEvent(cs, stageFree, 0));
    ck(cudaEventRecord(sendBeg_[s], cs));
    ok = ok && st == nvcompSuccess
            && xfer_pack(k.slotPtrs, k.compBytes, n_, k.hdr, hHdr_[s], hData_[s], cs) == cudaSuccess;
    // 終わりの印は失敗しても記録する（待つ側が止まらないように）
    ck(cudaEventRecord(encEnd_[s], cs));
    ck(cudaEventRecord(k.done, cs));
    k.used = true;
    used_[s] = true;
    if (!ok) qWarning() << "[xfer] 圧縮に失敗:" << int(st) << cudaGetErrorString(cudaGetLastError());
    cudaSetDevice(dst_);
    return ok;
}

// 展開（デコード・エンコード共通。書き込み先とストリームは呼び出し側が渡す）
bool XferCodec::decompress(int s, int ring, cudaStream_t st, const std::vector<Dst>& dst)
{
    if (!ready_) return false;
    const int o = ringSize_ > 0 ? ring : 0;   // 展開先
    if (o < 0 || o >= int(dTight_.size())) return false;
    if (ringSize_ == 0 && dst.size() * 2 != planes_.size()) return false;
    bool ok = true;
    auto ck = [&](cudaError_t e) { if (e != cudaSuccess) ok = false; };
    ck(cudaSetDevice(dst_));
    ck(cudaEventRecord(decBeg_[s], st));
    // 取り込み: 見込み（直近の実績 × 1.25 + 1MB。最初は全体の 1/4）までは DMA、超えた分はカーネル。外れても速さだけの問題
    //   ヘッダ（チャンクの大きさ・位置）はカーネルが検査し、壊れていたら hFlag_[1]
    const size_t cap = maxComp_ * size_t(n_);
    const size_t est = lastComp_ ? lastComp_ + lastComp_ / 4 + (size_t(1) << 20) : cap / 4;
    ck(xfer_fetch(hHdr_[s], dHdr_, n_, hData_[s], dData_, std::min(est, cap), maxComp_, cap,
                  (const void**)dCompPtrs_, dCompBytes_, &hFlag_[1], st));
    ck(cudaEventRecord(fetchEnd_[s], st));
    nvcompStatus_t r = nvcompSuccess;
    switch (algo_) {
    case Cascaded:
        r = nvcompBatchedCascadedDecompressAsync(dCompPtrs_, dCompBytes_, dOutBufBytes_, dOutBytes_, n_, dTemp_, dTempBytes_,
                                                 dOutPtrs_[o], cascadedDOpts(bps_), (nvcompStatus_t*)dStatus_, st);
        break;
    case Bitcomp: {
        const bitcompHandle_t h = static_cast<bitcompHandle_t>(dPlan_);
        r = (bitcompSetStream(h, st) == BITCOMP_SUCCESS && bitcompUncompress(h, dData_, dTight_[o]) == BITCOMP_SUCCESS)
          ? nvcompSuccess : nvcompErrorInternal;
        break;
    }
    case ANS:
        r = nvcompBatchedANSDecompressAsync(dCompPtrs_, dCompBytes_, dOutBufBytes_, dOutBytes_, n_, dTemp_, dTempBytes_,
                                            dOutPtrs_[o], nvcompBatchedANSDecompressDefaultOpts, (nvcompStatus_t*)dStatus_, st);
        break;
    default: r = nvcompErrorInvalidValue; break;
    }
    ok = ok && r == nvcompSuccess;
    // チャンクごとの結果と展開後の大きさを GPU 上で検査して hFlag_[1] へ（Bitcomp は状態を返さない）
    if (r == nvcompSuccess && algo_ != Bitcomp)
        ck(xfer_status((const int*)dStatus_, n_, dOutBytes_, dOutBufBytes_, &hFlag_[1], st));
    ck(cudaEventRecord(decompEnd_[s], st));
    // 差分戻し（詰めたまま渡すときは GUI の変換カーネルが戻すので、ここでは何もしない）
    for (size_t t = 0; ok && ringSize_ == 0 && t < dst.size(); t++) {
        const Plane& py = planes_[2 * t];
        const Plane& pu = planes_[2 * t + 1];
        ok = xfer_inv(dTight_[0] + py.off, dst[t].y,  dst[t].yPitch,  py.elems, py.rows, bps_, py.lag, delta_, st) == cudaSuccess
          && xfer_inv(dTight_[0] + pu.off, dst[t].uv, dst[t].uvPitch, pu.elems, pu.rows, bps_, pu.lag, delta_, st) == cudaSuccess;
    }
    ck(cudaEventRecord(decEnd_[s], st));
    if (!ok) qWarning() << "[xfer] 展開に失敗:" << int(r) << cudaGetErrorString(cudaGetLastError());
    return ok;
}

// GPU 上で見つかった失敗（nvCOMP のチャンクの失敗・展開後の大きさの不一致・ヘッダの破損）。
//   GPU が書くのは後なので、数フレーム遅れて分かる。clear なら読んだ後で戻す
bool XferCodec::deviceError(bool clear)
{
    if (!hFlag_) return false;
    volatile int* f = hFlag_;
    const bool e = f[0] != 0 || f[1] != 0;
    if (e) {
        qWarning() << "[xfer] GPU 上で失敗を検出（圧縮" << f[0] << "/ 展開" << f[1] << "）";
        if (clear) { f[0] = 0; f[1] = 0; }
    }
    return e;
}

bool XferCodec::takeStats(int s, Stats& out)
{
    if (!ready_ || !used_[s]) return false;
    if (cudaEventQuery(encEnd_[s]) != cudaSuccess || cudaEventQuery(decEnd_[s]) != cudaSuccess) { cudaGetLastError(); return false; }
    float c = 0, w = 0, d = 0;
    if (cudaEventElapsedTime(&c, encBeg_[s], cmpEnd_[s]) != cudaSuccess ||
        cudaEventElapsedTime(&w, sendBeg_[s], encEnd_[s]) != cudaSuccess ||
        cudaEventElapsedTime(&d, decBeg_[s], decEnd_[s]) != cudaSuccess) { cudaGetLastError(); return false; }
    out = Stats{};
    out.compressMs = c;
    out.sendMs     = w;
    out.expandMs   = d;
    out.rawBytes   = double(total_);
    out.compBytes  = double(hHdr_[s][1]);   // xfer_pack が書いたヘッダの合計バイト
    lastComp_      = size_t(hHdr_[s][1]);   // 次の取り込みの見込みに使う
    float f = 0, x = 0;
    if (cudaEventElapsedTime(&f, decBeg_[s], fetchEnd_[s]) == cudaSuccess &&
        cudaEventElapsedTime(&x, fetchEnd_[s], decompEnd_[s]) == cudaSuccess) {
        out.fetchMs  = f;
        out.decompMs = x;
        out.invMs    = (std::max)(0.0f, d - f - x);
    } else {
        cudaGetLastError();
    }
    return true;
}

void XferCodec::release()
{
    auto freeDev = [](auto*& p) { if (p) cudaFree((void*)p); p = nullptr; };
    if (src_ >= 0) {
        cudaSetDevice(src_);
        cudaDeviceSynchronize();
        freeDev(sInBytes_);
        for (SrcSet& k : sets_) {
            freeDev(k.tight); freeDev(k.inPtrs); freeDev(k.slotBuf); freeDev(k.slotPtrs);
            freeDev(k.compBytes); freeDev(k.statuses); freeDev(k.hdr); freeDev(k.temp);
            if (k.bcPlan) { bitcompDestroyPlan(static_cast<bitcompHandle_t>(k.bcPlan)); k.bcPlan = nullptr; }
            if (k.ready)  { cudaEventDestroy(k.ready); k.ready = nullptr; }
            if (k.done)   { cudaEventDestroy(k.done); k.done = nullptr; }
            if (k.stream) { cudaStreamDestroy(k.stream); k.stream = nullptr; }
            k.used = false;
        }
        destroyEvents(encBeg_); destroyEvents(cmpEnd_); destroyEvents(sendBeg_); destroyEvents(encEnd_);
    }
    if (dst_ >= 0) {
        cudaSetDevice(dst_);
        cudaDeviceSynchronize();
        freeDev(dData_); freeDev(dCompPtrs_); freeDev(dCompBytes_);
        for (uint8_t*& p : dTight_)   freeDev(p);
        for (void**&   p : dOutPtrs_) freeDev(p);
        dTight_.clear(); dOutPtrs_.clear();
        freeDev(dOutBufBytes_); freeDev(dOutBytes_); freeDev(dStatus_); freeDev(dTemp_); freeDev(dHdr_);
        if (dPlan_) { bitcompDestroyPlan(static_cast<bitcompHandle_t>(dPlan_)); dPlan_ = nullptr; }
        destroyEvents(decBeg_); destroyEvents(fetchEnd_); destroyEvents(decompEnd_); destroyEvents(decEnd_);
    }
    if (hFlag_) { cudaFreeHost(hFlag_); hFlag_ = nullptr; }
    for (uint64_t* p : hHdr_)  if (p) cudaFreeHost(p);
    for (uint8_t*  p : hData_) if (p) cudaFreeHost(p);
    hHdr_.clear(); hData_.clear();
    used_.clear();
    sTempBytes_ = dTempBytes_ = 0;
    ready_ = false;
}
