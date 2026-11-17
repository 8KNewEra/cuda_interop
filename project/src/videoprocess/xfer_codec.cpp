#include "src/videoprocess/xfer_codec.h"

#include <QDebug>
#include <algorithm>

#include "nvcomp/cascaded.h"   // 3rdparty/nvcomp/include（NVIDIA nvCOMP SDK）
#include "nvcomp/bitcomp.h"
#include "nvcomp/ans.h"

// cuda_kernels/xfer_compress.cu
extern "C" cudaError_t xfer_fwd(const uint8_t* src, size_t srcPitch, uint8_t* dst, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st);
extern "C" cudaError_t xfer_inv(const uint8_t* src, uint8_t* dst, size_t dstPitch, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st);
extern "C" cudaError_t xfer_pack(const void* const* slots, const size_t* compBytes, int n,
                                 uint64_t* devHdr, uint64_t* hostHdr, uint8_t* hostData, cudaStream_t st);
extern "C" cudaError_t xfer_fetch(const uint64_t* hostHdr, uint64_t* devHdr, int n, const uint8_t* hostData,
                                  uint8_t* devData, size_t copyBytes, const void** compPtrs, size_t* compBytes,
                                  cudaStream_t st);

namespace {
size_t align16(size_t v) { return (v + 15) & ~size_t(15); }

// ---- アルゴリズムごとの nvCOMP の呼び分け ----
//   データ型: NV12 は 1 バイト、P010 は 2 バイト（ANS はバイト列のみ）
nvcompBatchedCascadedCompressOpts_t cascadedOpts(int bps)
{
    nvcompBatchedCascadedCompressOpts_t o = nvcompBatchedCascadedCompressDefaultOpts;
    o.common_opts.data_type = (bps == 2) ? NVCOMP_TYPE_USHORT : NVCOMP_TYPE_UCHAR;
    return o;
}
nvcompBatchedCascadedDecompressOpts_t cascadedDOpts(int bps)
{
    nvcompBatchedCascadedDecompressOpts_t o = nvcompBatchedCascadedDecompressDefaultOpts;
    o.common_opts = cascadedOpts(bps).common_opts;   // 圧縮と同じ条件に絞る（展開が速くなる）
    return o;
}
nvcompBatchedBitcompCompressOpts_t bitcompOpts(int bps)
{
    nvcompBatchedBitcompCompressOpts_t o = nvcompBatchedBitcompCompressDefaultOpts;
    o.data_type = (bps == 2) ? NVCOMP_TYPE_USHORT : NVCOMP_TYPE_UCHAR;
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
                     const std::vector<int>& tileW, const std::vector<int>& tileH, int nStage)
{
    release();
    src_ = srcGpu; dst_ = dstGpu; algo_ = algo; delta_ = delta; bps_ = bps; nStage_ = nStage;
    if (algo == None || tileW.empty() || tileW.size() != tileH.size()) return false;

    // 詰めた並び: タイルごとに Y（w × h、差は 1 つ左）、UV（w × h/2、U と V が交互なので差は 2 つ左）
    planes_.clear();
    size_t off = 0;
    for (size_t t = 0; t < tileW.size(); t++) {
        const int w = tileW[t], h = tileH[t];
        planes_.push_back({ off, w, h, 1 });
        off = align16(off + size_t(w) * h * bps);
        planes_.push_back({ off, w, h / 2, 2 });
        off = align16(off + size_t(w) * (h / 2) * bps);
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
        st = nvcompBatchedBitcompCompressGetMaxOutputChunkSize(chunk_, bitcompOpts(bps_), &maxc);
        if (st == nvcompSuccess) st = nvcompBatchedBitcompCompressGetTempSize(n_, chunk_, bitcompOpts(bps_), &ctemp, total_, 0);
        if (st == nvcompSuccess) st = nvcompBatchedBitcompDecompressGetTempSize(n_, chunk_, nvcompBatchedBitcompDecompressDefaultOpts, &dtemp, total_, 0);
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
    for (int i = 0; i < n_; i++) chunkBytes[i] = std::min(chunk_, total_ - size_t(i) * chunk_);

    bool ok = true;
    auto chk = [&](cudaError_t e) { if (e != cudaSuccess) ok = false; };

    // ---- デコード側 GPU ----
    cudaSetDevice(src_);
    chk(cudaMalloc(&sInBytes_, sizeof(size_t) * n_));
    if (ok) chk(cudaMemcpy(sInBytes_, chunkBytes.data(), sizeof(size_t) * n_, cudaMemcpyHostToDevice));
    sTempBytes_ = ctemp;
    nextSet_ = 0;
    for (SrcSet& k : sets_) {
        k.used = false;
        chk(cudaStreamCreateWithFlags(&k.stream, cudaStreamNonBlocking));
        chk(cudaEventCreateWithFlags(&k.ready, cudaEventDisableTiming));
        chk(cudaEventCreateWithFlags(&k.done, cudaEventDisableTiming));
        chk(cudaMalloc(&k.tight, total_));
        chk(cudaMalloc(&k.inPtrs, sizeof(void*) * n_));
        chk(cudaMalloc(&k.slotBuf, maxComp_ * n_));
        chk(cudaMalloc(&k.slotPtrs, sizeof(void*) * n_));
        chk(cudaMalloc(&k.compBytes, sizeof(size_t) * n_));
        chk(cudaMalloc(&k.statuses, sizeof(nvcompStatus_t) * n_));
        chk(cudaMalloc(&k.hdr, sizeof(uint64_t) * (2 + 2 * size_t(n_))));
        if (ctemp) chk(cudaMalloc(&k.temp, ctemp));
        if (!ok) break;
        std::vector<void*> in(n_), slot(n_);
        for (int i = 0; i < n_; i++) { in[i] = k.tight + size_t(i) * chunk_; slot[i] = k.slotBuf + size_t(i) * maxComp_; }
        chk(cudaMemcpy(k.inPtrs, in.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
        chk(cudaMemcpy(k.slotPtrs, slot.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
    }
    encBeg_.assign(nStage_, nullptr); encEnd_.assign(nStage_, nullptr); used_.assign(nStage_, false);
    cmpEnd_.assign(nStage_, nullptr); sendBeg_.assign(nStage_, nullptr);
    for (int s = 0; s < nStage_; s++) {
        chk(cudaEventCreate(&encBeg_[s])); chk(cudaEventCreate(&encEnd_[s]));
        chk(cudaEventCreate(&cmpEnd_[s])); chk(cudaEventCreate(&sendBeg_[s]));
    }

    // ---- メインメモリ（Portable: どの GPU からも使える、Mapped: カーネルから直接読み書きできる）----
    hHdr_.assign(nStage_, nullptr); hData_.assign(nStage_, nullptr);
    for (int s = 0; s < nStage_; s++) {
        chk(cudaHostAlloc((void**)&hHdr_[s], sizeof(uint64_t) * (2 + 2 * size_t(n_)), cudaHostAllocPortable | cudaHostAllocMapped));
        chk(cudaHostAlloc((void**)&hData_[s], maxComp_ * n_, cudaHostAllocPortable | cudaHostAllocMapped));
    }

    // ---- 表示用 GPU ----
    cudaSetDevice(dst_);
    chk(cudaMalloc(&dData_, maxComp_ * n_));
    chk(cudaMalloc(&dHdr_, sizeof(uint64_t) * (2 + 2 * size_t(n_))));
    lastComp_ = 0;
    chk(cudaMalloc(&dCompPtrs_, sizeof(void*) * n_));
    chk(cudaMalloc(&dCompBytes_, sizeof(size_t) * n_));
    chk(cudaMalloc(&dTight_, total_));
    chk(cudaMalloc(&dOutPtrs_, sizeof(void*) * n_));
    chk(cudaMalloc(&dOutBufBytes_, sizeof(size_t) * n_));
    chk(cudaMalloc(&dOutBytes_, sizeof(size_t) * n_));
    chk(cudaMalloc(&dStatus_, sizeof(nvcompStatus_t) * n_));
    if (dtemp) { chk(cudaMalloc(&dTemp_, dtemp)); dTempBytes_ = dtemp; }
    if (ok) {
        std::vector<void*> out(n_);
        for (int i = 0; i < n_; i++) out[i] = dTight_ + size_t(i) * chunk_;
        chk(cudaMemcpy(dOutPtrs_, out.data(), sizeof(void*) * n_, cudaMemcpyHostToDevice));
        chk(cudaMemcpy(dOutBufBytes_, chunkBytes.data(), sizeof(size_t) * n_, cudaMemcpyHostToDevice));
    }
    decBeg_.assign(nStage_, nullptr); decEnd_.assign(nStage_, nullptr);
    fetchEnd_.assign(nStage_, nullptr); decompEnd_.assign(nStage_, nullptr);
    for (int s = 0; s < nStage_; s++) {
        chk(cudaEventCreate(&decBeg_[s]));   chk(cudaEventCreate(&decEnd_[s]));
        chk(cudaEventCreate(&fetchEnd_[s])); chk(cudaEventCreate(&decompEnd_[s]));
    }

    if (!ok) {
        qWarning() << "[xfer] メモリの確保に失敗:" << cudaGetErrorString(cudaGetLastError());
        release();
        return false;
    }
    ready_ = true;
    qDebug().noquote() << QString("[xfer] GPU%1 → GPU%2 を %3 で圧縮転送（%4 チャンク × %5 KB、1 フレーム %6 MB%7）")
                              .arg(src_).arg(dst_).arg(name(algo_)).arg(n_).arg(chunk_ / 1024)
                              .arg(total_ / 1e6, 0, 'f', 1).arg(delta_ ? "、差分あり" : "");
    return true;
}

bool XferCodec::encode(int s, const std::vector<const AVFrame*>& frames, cudaEvent_t stageFree)
{
    if (!ready_ || frames.size() * 2 != planes_.size()) return false;
    cudaSetDevice(src_);
    bool ok = true;
    SrcSet& k = sets_[nextSet_];
    nextSet_ = (nextSet_ + 1) % kSets;
    // この組の前回（kSets 個前のフレーム）の完了を既定ストリームに待たせる。
    //   以降に積まれる cuvid の出力コピーが、そのフレームの面を上書きする前に読み終えていることの保証
    //   （面はリングが一周するまで残るので、実際にはまず待たない）
    if (k.used) cudaStreamWaitEvent(0, k.done, 0);
    // 既定ストリーム = cuvid の出力コピーの後。ここからは組のストリームで進める
    cudaEventRecord(k.ready, 0);
    cudaStreamWaitEvent(k.stream, k.ready, 0);
    cudaStream_t cs = k.stream;
    cudaEventRecord(encBeg_[s], cs);
    for (size_t t = 0; t < frames.size(); t++) {
        const AVFrame* f = frames[t];
        const Plane& py = planes_[2 * t];
        const Plane& pu = planes_[2 * t + 1];
        ok = ok && xfer_fwd(f->data[0], size_t(f->linesize[0]), k.tight + py.off, py.elems, py.rows, bps_, py.lag, delta_, cs) == cudaSuccess
                && xfer_fwd(f->data[1], size_t(f->linesize[1]), k.tight + pu.off, pu.elems, pu.rows, bps_, pu.lag, delta_, cs) == cudaSuccess;
    }
    nvcompStatus_t st = nvcompSuccess;
    switch (algo_) {
    case Cascaded:
        st = nvcompBatchedCascadedCompressAsync(k.inPtrs, sInBytes_, chunk_, n_, k.temp, sTempBytes_,
                                               k.slotPtrs, k.compBytes, cascadedOpts(bps_), (nvcompStatus_t*)k.statuses, cs);
        break;
    case Bitcomp:
        st = nvcompBatchedBitcompCompressAsync(k.inPtrs, sInBytes_, chunk_, n_, k.temp, sTempBytes_,
                                              k.slotPtrs, k.compBytes, bitcompOpts(bps_), (nvcompStatus_t*)k.statuses, cs);
        break;
    case ANS:
        st = nvcompBatchedANSCompressAsync(k.inPtrs, sInBytes_, chunk_, n_, k.temp, sTempBytes_,
                                          k.slotPtrs, k.compBytes, ansOpts(), (nvcompStatus_t*)k.statuses, cs);
        break;
    default: st = nvcompErrorInvalidValue; break;
    }
    cudaEventRecord(cmpEnd_[s], cs);
    // メインメモリの枠 s を表示用 GPU が読み終えるまで待ってから書く
    if (stageFree) cudaStreamWaitEvent(cs, stageFree, 0);
    cudaEventRecord(sendBeg_[s], cs);
    ok = ok && st == nvcompSuccess
            && xfer_pack(k.slotPtrs, k.compBytes, n_, k.hdr, hHdr_[s], hData_[s], cs) == cudaSuccess;
    cudaEventRecord(encEnd_[s], cs);
    cudaEventRecord(k.done, cs);
    k.used = true;
    used_[s] = true;
    cudaSetDevice(dst_);
    if (!ok) qWarning() << "[xfer] 圧縮に失敗:" << int(st) << cudaGetErrorString(cudaGetLastError());
    return ok;
}

bool XferCodec::decode(int s, cudaStream_t st, const std::vector<Dst>& dst)
{
    if (!ready_ || dst.size() * 2 != planes_.size()) return false;
    cudaSetDevice(dst_);
    bool ok = true;
    cudaEventRecord(decBeg_[s], st);
    // 取り込み: 圧縮後の大きさの見込み（少し前のフレームの実績 × 1.25 + 1MB）までは DMA、超えた分はカーネル
    //   最初のフレームは実績が無いので全体の 1/4 を見込む（外れても正しさは変わらない。速さだけ）
    const size_t cap = maxComp_ * size_t(n_);
    const size_t est = lastComp_ ? lastComp_ + lastComp_ / 4 + (size_t(1) << 20) : cap / 4;
    ok = xfer_fetch(hHdr_[s], dHdr_, n_, hData_[s], dData_, std::min(est, cap),
                    (const void**)dCompPtrs_, dCompBytes_, st) == cudaSuccess;
    cudaEventRecord(fetchEnd_[s], st);
    nvcompStatus_t r = nvcompSuccess;
    switch (algo_) {
    case Cascaded:
        r = nvcompBatchedCascadedDecompressAsync(dCompPtrs_, dCompBytes_, dOutBufBytes_, dOutBytes_, n_, dTemp_, dTempBytes_,
                                                 dOutPtrs_, cascadedDOpts(bps_), (nvcompStatus_t*)dStatus_, st);
        break;
    case Bitcomp:
        r = nvcompBatchedBitcompDecompressAsync(dCompPtrs_, dCompBytes_, dOutBufBytes_, dOutBytes_, n_, dTemp_, dTempBytes_,
                                                dOutPtrs_, nvcompBatchedBitcompDecompressDefaultOpts, (nvcompStatus_t*)dStatus_, st);
        break;
    case ANS:
        r = nvcompBatchedANSDecompressAsync(dCompPtrs_, dCompBytes_, dOutBufBytes_, dOutBytes_, n_, dTemp_, dTempBytes_,
                                            dOutPtrs_, nvcompBatchedANSDecompressDefaultOpts, (nvcompStatus_t*)dStatus_, st);
        break;
    default: r = nvcompErrorInvalidValue; break;
    }
    ok = ok && r == nvcompSuccess;
    cudaEventRecord(decompEnd_[s], st);
    for (size_t t = 0; ok && t < dst.size(); t++) {
        const Plane& py = planes_[2 * t];
        const Plane& pu = planes_[2 * t + 1];
        ok = xfer_inv(dTight_ + py.off, dst[t].y,  dst[t].yPitch,  py.elems, py.rows, bps_, py.lag, delta_, st) == cudaSuccess
          && xfer_inv(dTight_ + pu.off, dst[t].uv, dst[t].uvPitch, pu.elems, pu.rows, bps_, pu.lag, delta_, st) == cudaSuccess;
    }
    cudaEventRecord(decEnd_[s], st);
    if (!ok) qWarning() << "[xfer] 展開に失敗:" << int(r) << cudaGetErrorString(cudaGetLastError());
    return ok;
}

bool XferCodec::takeStats(int s, double& cmpMs, double& sendMs, double& decMs, double& rawBytes, double& compBytes,
                          double* fetchMs, double* decompMs, double* invMs)
{
    if (!ready_ || !used_[s]) return false;
    if (cudaEventQuery(encEnd_[s]) != cudaSuccess || cudaEventQuery(decEnd_[s]) != cudaSuccess) { cudaGetLastError(); return false; }
    float c = 0, w = 0, d = 0;
    if (cudaEventElapsedTime(&c, encBeg_[s], cmpEnd_[s]) != cudaSuccess ||
        cudaEventElapsedTime(&w, sendBeg_[s], encEnd_[s]) != cudaSuccess ||
        cudaEventElapsedTime(&d, decBeg_[s], decEnd_[s]) != cudaSuccess) { cudaGetLastError(); return false; }
    cmpMs = c; sendMs = w; decMs = d;
    rawBytes  = double(total_);
    compBytes = double(hHdr_[s][1]);   // xfer_pack が書いたヘッダ（合計バイト）
    lastComp_ = size_t(hHdr_[s][1]);   // 次の取り込みの見込みに使う
    // 展開の内訳（取り込み / nvCOMP の展開 / 差分を戻して書く）
    float f = 0, x = 0;
    if (cudaEventElapsedTime(&f, decBeg_[s], fetchEnd_[s]) == cudaSuccess &&
        cudaEventElapsedTime(&x, fetchEnd_[s], decompEnd_[s]) == cudaSuccess) {
        if (fetchMs)  *fetchMs  = f;
        if (decompMs) *decompMs = x;
        if (invMs)    *invMs    = (std::max)(0.0f, d - f - x);
    } else {
        cudaGetLastError();
        if (fetchMs) *fetchMs = -1;
        if (decompMs) *decompMs = -1;
        if (invMs) *invMs = -1;
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
            if (k.ready)  { cudaEventDestroy(k.ready); k.ready = nullptr; }
            if (k.done)   { cudaEventDestroy(k.done); k.done = nullptr; }
            if (k.stream) { cudaStreamDestroy(k.stream); k.stream = nullptr; }
            k.used = false;
        }
        for (cudaEvent_t& e : encBeg_) if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : encEnd_) if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : cmpEnd_) if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : sendBeg_) if (e) { cudaEventDestroy(e); e = nullptr; }
    }
    if (dst_ >= 0) {
        cudaSetDevice(dst_);
        cudaDeviceSynchronize();
        freeDev(dData_); freeDev(dCompPtrs_); freeDev(dCompBytes_); freeDev(dTight_); freeDev(dOutPtrs_);
        freeDev(dOutBufBytes_); freeDev(dOutBytes_); freeDev(dStatus_); freeDev(dTemp_); freeDev(dHdr_);
        for (cudaEvent_t& e : fetchEnd_)  if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : decompEnd_) if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : decBeg_) if (e) { cudaEventDestroy(e); e = nullptr; }
        for (cudaEvent_t& e : decEnd_) if (e) { cudaEventDestroy(e); e = nullptr; }
    }
    for (uint64_t*& p : hHdr_)  if (p) { cudaFreeHost(p); p = nullptr; }
    for (uint8_t*&  p : hData_) if (p) { cudaFreeHost(p); p = nullptr; }
    sTempBytes_ = dTempBytes_ = 0;
    ready_ = false;
}
