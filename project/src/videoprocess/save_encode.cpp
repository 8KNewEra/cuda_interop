#include "src/videoprocess/save_encode.h"
#include "qdebug.h"
#include <cmath>

// ======================================================================
// Windows: スレッド優先度をフォアグラウンド状態に依存させないための設定
//
//  Windows は既定で「フォアグラウンドプロセスのスレッド」に優先度ブーストと
//  3倍のタイムクォンタムを与える (Win32PrioritySeparation)。
//  そのため別ウィンドウを前面に出すとエンコードスレッドが相対的に痩せ、
//  fps が落ちる。ここで明示的に優先度を固定して依存を切る。
// ======================================================================
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <avrt.h>
  #ifdef _MSC_VER
    #pragma comment(lib, "avrt.lib")   // MinGW の場合は .pro に LIBS += -lavrt
  #endif

namespace {

// 映像エンコードワーカーの優先度。
// UI や音声が引っ張られるようなら THREAD_PRIORITY_HIGHEST に落として比較する。
const int kEncoderThreadPriority = THREAD_PRIORITY_TIME_CRITICAL;
const int kAudioThreadPriority   = THREAD_PRIORITY_ABOVE_NORMAL;

// MMCSS(Multimedia Class Scheduler Service) 登録の RAII ヘルパ。
// 登録されたスレッドはフォアグラウンド/バックグラウンドに関係なく
// マルチメディア用のスケジューリング保証を受ける。
struct MmcssScope
{
    HANDLE h = nullptr;

    MmcssScope(const wchar_t* task, AVRT_PRIORITY prio)
    {
        DWORD taskIndex = 0;
        h = AvSetMmThreadCharacteristicsW(task, &taskIndex);
        if (h) {
            AvSetMmThreadPriority(h, prio);
        } else {
            qDebug() << "[save_encode] AvSetMmThreadCharacteristics failed:"
                     << (int)GetLastError();
        }
    }

    ~MmcssScope()
    {
        if (h) AvRevertMmThreadCharacteristics(h);
    }

    MmcssScope(const MmcssScope&)            = delete;
    MmcssScope& operator=(const MmcssScope&) = delete;
};

} // namespace
#endif // _WIN32


// ==============================================================
// ★ NVENC 入力サーフェスの占有判定
//
//   FFmpeg の nvenc は AV_PIX_FMT_CUDA の入力フレームを
//   av_frame_ref() で保持し、対応する出力パケットを処理した時点で
//   av_frame_unref() する。
//   したがって「バッファの参照カウントが 1（＝こちらのリングの分だけ）」
//   ならば、NVENC はもうこの面を読んでいない＝上書きして安全。
//
//   この判定は出力順・パケット数に一切依存しないので、
//   B-frame や 1フレーム複数パケットでも正しく効く。
// ==============================================================
namespace {

// UI 側が指定しうる B-frame の絶対上限
const int kMaxBFrames = 7;

// ★ nvenc_max_bframes() / nvenc_surface_estimate() / encode_ring_size() は
//   __global__.h へ移した。設定画面が同じ式で VRAM を見積もるため。

bool frame_is_unreferenced(const AVFrame* f)
{
    if (!f) return true;

    for (int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
        if (f->buf[i] && av_buffer_get_ref_count(f->buf[i]) > 1)
            return false;
    }
    return true;
}

} // namespace


// ==============================================================
// エラーの文言（画面に出す）
//   VRAM 不足とディスクの空き不足は、見て分かる文言にする
// ==============================================================
namespace {

QString av_err_text(int e)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, buf, sizeof(buf));
    return QString::fromLocal8Bit(buf);
}

// ファイルへの書き込みの失敗
QString write_error_text(int e)
{
    if (e == AVERROR(ENOSPC))
        return QStringLiteral("ディスクの空き不足");
    return QStringLiteral("書き込みエラー（%1）").arg(av_err_text(e));
}

[[noreturn]] void throw_text(const QString& s) { throw std::runtime_error(s.toUtf8().toStdString()); }

// CUDA の確保・作成の失敗は例外にする（コンストラクタの catch が解放する）
void cuda_check(cudaError_t e, const QString& what)
{
    if (e == cudaSuccess) return;
    cudaGetLastError();
    if (e == cudaErrorMemoryAllocation)
        throw_text(QStringLiteral("%1: VRAM 不足").arg(what));
    throw_text(QStringLiteral("%1: %2").arg(what, QString::fromUtf8(cudaGetErrorString(e))));
}

} // namespace


// ==============================================================
// 生成と破棄
//   コンストラクタの途中で失敗（VRAM 不足、エンコーダを開けない、出力ファイルを開けない等）したら、
//   それまでに確保したものを release_all で全部返してから例外を投げ直す（呼び出し側は catch してエラー表示。
//   返さないと、失敗するたびに VRAM・CUDA のストリーム・イベント・FFmpeg のコンテキストが残り続ける）
//   init の途中のどこで止まっても release_all が後始末できるよう、メンバは全て nullptr / 空から始め、確保したらすぐメンバへ置く
// ==============================================================
save_encode::save_encode(int h,int w,bool srcRgba16_) {
    try {
        init(h, w, srcRgba16_);
    } catch (...) {
        qWarning() << "[save_encode] 初期化に失敗したので確保したものを解放する";
        stop_encoder_threads();   // 起動前なら何もしない
        stop_audio_thread();
        release_all(false);
        throw;
    }
}

void save_encode::init(int h,int w,bool srcRgba16_) {
    //最初にGPU設定
    cuda_check(cudaSetDevice(g_openglDeviceID), "GPU の選択");

    // 入力の形式と出力のビット深度（10bit は P010。設定画面と同じ判定 encode_out_10bit）
    srcRgba16 = srcRgba16_;
    out10     = encode_out_10bit(encodeSettings, srcRgba16);
    qDebug() << "[save_encode] 入力" << (srcRgba16 ? "RGBA16（10bit 経路）" : "RGBA8（8bit 経路）")
             << " 出力" << (out10 ? "10bit（P010）" : "8bit（NV12）");

    // ★ 画像処理インスタンスの生成
    //   これが無いと encode_video() 内の
    //   CUDA_IMG_Proc->encode_rows_*() が nullptr 参照になる。
    //   メンバを触らない実装だと「たまたま動く」ため気づきにくい。
    if (!CUDA_IMG_Proc) {
        CUDA_IMG_Proc = new CUDA_ImageProcess();
    }

    // 書き出す大きさは偶数に切り下げる（4:2:0。奇数だと NVENC が偶数に切り上げ、余った列・行に未初期化の値が入る）。
    //   右端 1 列・下端 1 行を捨てる。テクスチャは元の大きさのまま読む（上下反転は元の高さで。texW_ / texH_）
    texW_ = w;
    texH_ = h;
    width_  = w & ~1;
    height_ = h & ~1;
    if (width_ != w || height_ != h)
        qDebug() << "[save_encode] 幅・高さが奇数なので" << w << "x" << h << "→" << width_ << "x" << height_ << "で書き出す";
    frame_index = 0;
    int ret = 0;

    // ★ 出力の形は最初に確定させる。下のログも分岐も参照する。
    separate_ = encodeSettings.separate_files;

    // ------------------------------------------------------------------
    // ★ B-frame から surfaces / async_depth / ring 段数を決める
    //
    //   【デッドロックの構造】
    //   NVENC は「入力サーフェスを nvenc_surfaces 枚まで受け取り、
    //   出力の準備ができた段階で packet を返す」。B-frame があると
    //   frameIntervalP(= b_frames+1) 段ぶんリオーダのために入力を保持し、
    //   さらに async_depth 段ぶん出力キューに溜めてから返し始める。
    //   その間 AVFrame の参照は返ってこない。
    //
    //   ここでこちらの ring が NVENC の入力プールより小さいと、
    //   NVENC が出力を開始する前にこちらが空き slot 待ちで止まる。
    //   （実測: b_frames=3 で submitted=11 / completed=0 のまま停止）
    //
    //   【解決】不変条件 ring > nvenc_surfaces を必ず満たす。
    //   こうすれば、こちらが詰まる前に必ず NVENC の入力プールが満杯になり、
    //   NVENC は出力を出さざるを得なくなる（出さなければ ffmpeg 本体でも
    //   デッドロックするので、そういう状態は存在しない）。
    //
    //   FFmpeg は nb_surfaces を内部で
    //     FFMAX(4 + frameIntervalP, async_depth + frameIntervalP + 1, 指定値)
    //   のように決めるため、こちらから十分大きい surfaces を明示して
    //   「NVENC が最大何枚掴むか」を確定させてから ring を決める。
    //   （surfaces は出力ビットストリームバッファ等が主なので、
    //     入力が hw frame の場合は増やしてもVRAMをほとんど食わない）
    // ------------------------------------------------------------------
    //   ★ B-frame はまずエンコーダの上限で抑える。
    //     上限を超えると NVENC が出力を開始せず、ring をいくら増やしても
    //     completed=0 のまま停止する（nvenc_max_bframes のコメント参照）。
    const int codec_max_bf = nvenc_max_bframes(encodeSettings.codec);
    const int requested_bf = std::max(0, std::min(kMaxBFrames, encodeSettings.b_frames));

    b_frames_ = std::min(requested_bf, codec_max_bf);

    if (encodeSettings.b_frames != b_frames_) {
        qWarning() << "[save_encode] b_frames" << encodeSettings.b_frames
                   << "-> clamped to" << b_frames_
                   << " (codec =" << encodeSettings.codec
                   << ", NVENC max =" << codec_max_bf << ")";
    }

    const int frameIntervalP = b_frames_ + 1;

    // NVENC 内部のパイプライン段数。
    //   B-frame があるときはリオーダ自体がパイプラインになっており、
    //   こちらもワーカースレッドで並列化しているので ffmpeg 側で
    //   さらに出力を溜める必要はない。1 にして
    //   output_ready() の条件 (nb_ready + nb_pending >= async_depth) を
    //   即座に満たし、最初の packet が出るまでの枚数を減らす。
    //   ★ nvenc_surface_estimate() の中の async と必ず同じ値にすること。
    async_depth_ = (b_frames_ > 0) ? 1 : 4;

    // ★ 式は __global__.h（設定画面の VRAM 表示と共有）。別計算にしないこと。
    nvenc_surfaces_ = nvenc_surface_estimate(b_frames_);
    encodeRingSize  = encode_ring_size(b_frames_);

    // ---- 参考: リングが消費する VRAM を出しておく ----
    {
        const int wt = std::max(1, encodeSettings.width_tile);
        const int ht = std::max(1, encodeSettings.height_tile);
        const int tiles = std::max(1, encodeSettings.encode_tile);
        const double one_mb =
            (double)(width_ / wt) * (double)(height_ / ht) * 1.5 * (out10 ? 2 : 1) / (1024.0 * 1024.0);
        // ★ 設定画面の「想定使用 VRAM」と同じ式（encode_pool_mb。P010 は 1 サンプル 2 バイト）
        const int total_mb = encode_pool_mb(width_ / wt, height_ / ht,
                                            encodeRingSize, tiles, out10 ? 2 : 1);

        qDebug() << "[save_encode] 出力 =" << (separate_ ? "タイルごとに別ファイル" : "1ファイル")
                 << " files =" << (separate_ ? tiles : 1);
        qDebug() << "[save_encode] b_frames =" << b_frames_
                 << " async_depth =" << async_depth_
                 << " frameIntervalP =" << frameIntervalP
                 << " nvenc surfaces =" << nvenc_surfaces_
                 << " ring =" << encodeRingSize
                 << " / pool VRAM =" << total_mb << "MB"
                 << "(" << (encodeRingSize + 4) * tiles << "frames x"
                 << (int)one_mb << "MB )";

        // ★ レート制御とビットレート。ビットレートは「1 本あたり」なので、
        //   ファイル全体の合計はタイル数倍になる。両方出しておく。
        if (encodeSettings.rc_mode == "cqp") {
            qDebug() << "[save_encode] rc = constqp  qp =" << qBound(1, encodeSettings.cq, 51);
        } else if (encodeSettings.rc_mode == "cq") {
            qDebug() << "[save_encode] rc = vbr(CQ)  cq =" << qBound(1, encodeSettings.cq, 51)
                     << " 上限 1本あたり =" << encodeSettings.max_bit_rate / 1000000 << "Mbps"
                     << " / タイル" << tiles << "-> 合計上限 ="
                     << (qint64)encodeSettings.max_bit_rate * tiles / 1000000 << "Mbps";
        } else {
            qDebug() << "[save_encode] rc =" << encodeSettings.rc_mode
                     << " 1本あたり target ="
                     << encodeSettings.target_bit_rate / 1000000 << "Mbps  max ="
                     << encodeSettings.max_bit_rate / 1000000 << "Mbps"
                     << " / タイル" << tiles << "-> 合計 target ="
                     << (qint64)encodeSettings.target_bit_rate * tiles / 1000000 << "Mbps";
        }
    }

    // ★ 別ファイル構成では out.mp4 -> out.tile0.mp4 ... と拡張子の手前へ
    //   差し込む。音声はタイル 0 のファイルにだけ入れる。
    const auto open_fmt = [&](const QString& path) -> AVFormatContext* {
        AVFormatContext* f = nullptr;
        const int r = avformat_alloc_output_context2(
            &f, nullptr, nullptr, path.toUtf8().constData());
        if (r < 0 || !f) throw_text(QStringLiteral("出力形式を作れません（拡張子を確認）"));
        files_.push_back(f);
        return f;
    };

    const auto tile_path = [&](int i) {
        const QString& p = encodeSettings.encode_path;
        const int dot = p.lastIndexOf('.');
        const QString tag = QString(".tile%1").arg(i);
        return (dot < 0) ? (p + tag) : (p.left(dot) + tag + p.mid(dot));
    };

    if (!separate_)
        fmt_ctx = open_fmt(encodeSettings.encode_path);

    // ② Encoder / Stream を複数作る
    //   タイルは 8 まで（NV12Tiles の配列）。GPU の割り当て（tile_gpu_map）はタイル数ぶん要る（開始時に作る）
    if (encodeSettings.encode_tile < 1 || encodeSettings.encode_tile > 8 ||
        (int)encodeSettings.tile_gpu_map.size() < encodeSettings.encode_tile)
        throw_text(QStringLiteral("タイル数か GPU の割り当てが不正"));
    if (encodeSettings.encode_tile > 1 &&
        (width_ % (2 * encodeSettings.width_tile) != 0 || height_ % (2 * encodeSettings.height_tile) != 0))
        qWarning() << "[save_encode] 1 タイルの幅・高さが偶数になりません（" << width_ << "x" << height_
                   << "を" << encodeSettings.width_tile << "x" << encodeSettings.height_tile
                   << "分割）。書き出したファイルの大きさが変わり、色がにじみます";
    for (int i = 0; i < encodeSettings.encode_tile; i++) {
        ve.emplace_back(std::make_unique<VideoEncoder>());

        // ★ 書き込み先を先に決める。コーデックコンテキストの
        //   AVFMT_GLOBALHEADER 判定がこれを見るので、必ずその前に。
        if (separate_) {
            ve[i]->own_fmt = open_fmt(tile_path(i));
            if (i == 0) fmt_ctx = ve[i]->own_fmt;   // 音声はタイル 0 へ
        }
        ve[i]->fmt = separate_ ? ve[i]->own_fmt : fmt_ctx;

        //CUDAデバイスコンテキスト
        //  別 GPU のタイルは primary_ctx=1 で開く（Runtime API と同じコンテキストにして、
        //  こちらのストリームから AVFrame へ書けるようにする。transfer_remote の条件）
        const int dev = encodeSettings.tile_gpu_map[i];
        const QString gpuId = QString::number(dev);
        ve[i]->dev = dev;
        bool primaryCtx = false;
        if (dev != g_openglDeviceID) {
            AVDictionary* o = nullptr;
            av_dict_set(&o, "primary_ctx", "1", 0);
            ret = av_hwdevice_ctx_create(&ve[i]->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, gpuId.toUtf8().data(), o, 0);
            av_dict_free(&o);
            primaryCtx = (ret >= 0);
            if (!primaryCtx) {
                qWarning() << "[save_encode] GPU" << dev << "primary_ctx で開けないのでデバイス間コピーで送る:" << ret;
                if (ve[i]->hw_device_ctx) av_buffer_unref(&ve[i]->hw_device_ctx);
            }
        }
        if (!primaryCtx)
            ret = av_hwdevice_ctx_create(&ve[i]->hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
        if (ret < 0) throw_text(QStringLiteral("GPU %1 を開けません（%2）").arg(dev).arg(av_err_text(ret)));

        initialized_ffmpeg_hardware_context(i);   // ここで hw_frames.resize() される
        initialized_ffmpeg_codec_context(i,encodeSettings.encode_tile);

        // ★ stream 作成だけ
        ve[i]->stream = avformat_new_stream(ve[i]->fmt, nullptr);
        if (!ve[i]->stream) throw_text(QStringLiteral("メモリ不足（映像ストリーム）"));

        ve[i]->stream->time_base = ve[i]->codec_ctx->time_base;
        ve[i]->stream->avg_frame_rate = ve[i]->codec_ctx->framerate;
        ve[i]->stream->r_frame_rate   = ve[i]->codec_ctx->framerate;
        ret = avcodec_parameters_from_context(ve[i]->stream->codecpar, ve[i]->codec_ctx);
        if (ret < 0) throw_text(QStringLiteral("エンコーダの設定を書けません（%1）").arg(av_err_text(ret)));

        //Stream作成
        cuda_check(cudaStreamCreateWithFlags(&ve[i]->st, cudaStreamNonBlocking), "CUDA ストリームの作成");

        // 別 GPU のタイル: メインメモリ経由の転送の準備（圧縮するなら XferCodec も）。失敗したらデバイス間コピー
        const bool remote = (dev != g_openglDeviceID);
        if (remote && primaryCtx)
            ve[i]->remoteXfer = init_remote_xfer(*ve[i]);

        // 別 GPU のタイルの中継（変換の書き込み先）。圧縮転送するタイルは変換が詰めた並びへ直接書くので要らない
        if (remote && !ve[i]->codec) {
            const TileSize ts = tileSize();
            for (FrameSlot& fs : ve[i]->hw_frames) {
                cuda_check(cudaMallocPitch(&fs.d_y,  &fs.y_pitch,  ts.yBytes,  ts.h),     "転送の中継の確保");
                cuda_check(cudaMallocPitch(&fs.d_uv, &fs.uv_pitch, ts.uvBytes, ts.h / 2), "転送の中継の確保");
            }
        }

        //event作成（ready は記録するストリームの GPU に作る。transfer_remote なら別 GPU の rst）
        //  cudaEventBlockingSync は効果が無かったので付けない（スピン待ちのまま）
        if (ve[i]->remoteXfer) cudaSetDevice(dev);
        cudaError_t evErr = cudaSuccess;
        for(int j=0;j<encodeRingSize && evErr==cudaSuccess;j++){
            evErr = cudaEventCreateWithFlags(
                &ve[i]->hw_frames[j].ready,
                cudaEventDisableTiming
                );
        }
        cudaSetDevice(g_openglDeviceID);
        cuda_check(evErr, "CUDA イベントの作成");

        // ★ パイプライン用の初期化
        ve[i]->pkt = av_packet_alloc();
        if (!ve[i]->pkt) throw_text(QStringLiteral("メモリ不足（パケット）"));
        ve[i]->ring_capacity = encodeRingSize;
        ve[i]->submitted = 0;
        ve[i]->completed = 0;
    }

    //音声エンコーダー作成
    //   ★ 「音声を含める」を外しているときは音声ストリーム自体を作らない。
    audio_on_ = (VideoInfo.audio && encodeSettings.audio_enable);
    //   ★ スレッドはヘッダを書き終えてから起こす（下の avio_open などで例外になると、
    //     動いている std::thread が破棄されて std::terminate で落ちるため）
    if(audio_on_){
        init_audio_encoder();
    }

    // ③④ 全ファイルを開いてヘッダを書く
    //     ★ 必ず全 stream（映像 N 本 + 音声）を作り終えてから。
    for (int fi = 0; fi < (int)files_.size(); fi++) {
        AVFormatContext* f = files_[fi];
        const QString p = separate_ ? tile_path(fi) : encodeSettings.encode_path;

        ret = avio_open(&f->pb, p.toUtf8().constData(), AVIO_FLAG_WRITE);
        if (ret < 0) throw_text(QStringLiteral("出力ファイルを開けません（%1）").arg(av_err_text(ret)));
        openedPaths_.push_back(p);   // 失敗したときに消す

        ret = avformat_write_header(f, nullptr);
        if (ret < 0) throw_text(QStringLiteral("ヘッダを書けません（%1）").arg(write_error_text(ret)));
        headersWritten_ = fi + 1;
    }

    //Stream作成
    cuda_check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "CUDA ストリームの作成");

    //event作成
    cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "CUDA イベントの作成");

    // ⑤ ★ ヘッダを書き終えてからワーカー起動（mux 可能になってから）。スレッドの生成に失敗したら、コンストラクタの catch が止めてから解放する
    if(audio_on_){
        audioRunning = true;
        audioThread =
            std::thread(
                &save_encode::audio_loop,
                this
                );
    }
    start_encoder_threads();
}

save_encode::~save_encode() {
    close();
    qDebug() << "save_encode: Destructor called";
}

// 終了処理。呼び出し側は delete の前に呼び、結果（false なら failMessage）を画面に出す。2 回目以降は何もしない
bool save_encode::close()
{
    if (!closed_) {
        closed_ = true;
        // ==========================
        // ① 映像ワーカーを止める
        //    キュー末尾に eos を積むので、投入済みフレームは全部エンコードされ、
        //    NULL frame → EOF まで drain された上で join される
        // ==========================
        stop_encoder_threads();

        // ==========================
        // ② audioスレッド終了とflush処理
        // ==========================
        stop_audio_thread();

        release_all(true);
    }
    return !failed_;
}

// 途中の失敗を記録する（どのスレッドからでも。最初の 1 回の文言を残す）
void save_encode::fail(const QString& msg)
{
    std::lock_guard<std::mutex> lk(failMutex_);
    if (failed_) return;
    failMsg_ = msg;
    failed_ = true;
    qWarning() << "[save_encode] 失敗:" << msg;
}

// CUDA API の戻り値を見る（エンコード中。失敗したら fail して false）
bool save_encode::ck(cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    cudaGetLastError();
    fail(QStringLiteral("%1に失敗（%2）").arg(QString::fromUtf8(what), QString::fromUtf8(cudaGetErrorString(e))));
    return false;
}

QString save_encode::failMessage() const
{
    std::lock_guard<std::mutex> lk(failMutex_);
    return failMsg_;
}

// ==============================================================
// 全部の解放（デストラクタと、コンストラクタの失敗の両方から。スレッドは止めてから呼ぶ）
//   finalize = true : ファイルを書き終える（trailer）
//   finalize = false: 初期化の失敗。trailer は書かず、開いたファイルは消す（中身はヘッダまで）
//   途中まで確保した状態でも動くこと（nullptr / 空はとばす）
// ==============================================================
void save_encode::release_all(bool finalize)
{
    // ==========================
    // ③ ファイルの終了処理（mux 参加者が全員止まってから）
    // ==========================
    //   ★ 所有は files_ 1 箇所。fmt_ctx と ve[]->own_fmt は同じものを
    //     指すことがあるので、ここで畳んでから両方 null にする。
    for (int fi = 0; fi < (int)files_.size(); fi++) {
        AVFormatContext* f = files_[fi];
        if (!f) continue;
        if (f->pb) avio_flush(f->pb);

        // trailer はヘッダを書けたファイルだけ（書いていないファイルに書くと FFmpeg の中で落ちうる）
        if (finalize && fi < headersWritten_) {
            const int tret = av_write_trailer(f);
            qDebug() << "trailer ret =" << tret;
            if (tret < 0) fail(write_error_text(tret));
        }

        if (f->pb) {
            const int cret = avio_closep(&f->pb);   // 残りを書き出して閉じる（空き不足はここでも出る）
            if (finalize && cret < 0) fail(write_error_text(cret));
        }
        avformat_free_context(f);
    }
    files_.clear();
    if (!finalize)
        for (const QString& p : openedPaths_) QFile::remove(p);
    openedPaths_.clear();
    headersWritten_ = 0;
    fmt_ctx = nullptr;
    for (auto& e : ve) { if (e) { e->fmt = nullptr; e->own_fmt = nullptr; } }

    // ==========================
    // ④ 各メモリ解放
    //    ★ hw_frames.clear() は「全部触り終えた最後」に行う
    // ==========================
    for (size_t i = 0; i < ve.size(); i++) {

        //Streamの残処理を待つ
        if (ve[i]->st) {
            cudaStreamSynchronize(ve[i]->st);
        }
        if (ve[i]->rst) {
            cudaStreamSynchronize(ve[i]->rst);
        }

        for (int j = 0; j < (int)ve[i]->hw_frames.size(); j++) {

            //event削除（transfer_remote のタイルは別 GPU に作ってある）
            if (ve[i]->hw_frames[j].ready) {
                if (ve[i]->remoteXfer) cudaSetDevice(ve[i]->dev);
                cudaEventDestroy(ve[i]->hw_frames[j].ready);
                ve[i]->hw_frames[j].ready = nullptr;
                cudaSetDevice(g_openglDeviceID);
            }

            //中間バッファ解放（primary GPU の場合は AVFrame の中身を指しているので触らない）
            //  dev は GPU を決めた時点で入る（それより前に失敗したタイルは d_y も無い）
            if (ve[i]->dev != g_openglDeviceID) {
                if (ve[i]->hw_frames[j].d_y) {
                    cudaFree(ve[i]->hw_frames[j].d_y);
                    ve[i]->hw_frames[j].d_y = nullptr;
                }
                if (ve[i]->hw_frames[j].d_uv) {
                    cudaFree(ve[i]->hw_frames[j].d_uv);
                    ve[i]->hw_frames[j].d_uv = nullptr;
                }
            } else {
                ve[i]->hw_frames[j].d_y  = nullptr;
                ve[i]->hw_frames[j].d_uv = nullptr;
            }

            //ハードウェアフレームを解放
            if (ve[i]->hw_frames[j].frame) {
                av_frame_free(&ve[i]->hw_frames[j].frame);
                ve[i]->hw_frames[j].frame = nullptr;
            }
        }

        // ★ ここまで触り終えてから clear
        ve[i]->hw_frames.clear();
        release_remote_xfer(*ve[i]);

        //Stream削除
        if (ve[i]->st) {
            cudaStreamDestroy(ve[i]->st);
            ve[i]->st = nullptr;
        }

        //ハードウェアフレームコンテキストを解放
        if (ve[i]->hw_frames_ctx) {
            av_buffer_unref(&ve[i]->hw_frames_ctx);
            ve[i]->hw_frames_ctx = nullptr;
        }
        //コーデックコンテキストを解放
        if (ve[i]->codec_ctx) {
            avcodec_free_context(&ve[i]->codec_ctx);
            ve[i]->codec_ctx = nullptr;
        }
        //ハードウェアデバイスコンテキストを解放
        if (ve[i]->hw_device_ctx) {
            av_buffer_unref(&ve[i]->hw_device_ctx);
            ve[i]->hw_device_ctx = nullptr;
        }
        //パケット解放
        if (ve[i]->pkt) {
            av_packet_free(&ve[i]->pkt);
            ve[i]->pkt = nullptr;
        }
    }
    ve.clear();

    if (audio_enc_ctx){
        avcodec_free_context(&audio_enc_ctx);
    }
    if (swr_enc) {
        swr_free(&swr_enc);
    }
    if (audio_fifo) {
        av_audio_fifo_free(audio_fifo);
        audio_fifo = nullptr;
    }

    //Stream削除
    if(st){
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
        st=nullptr;
    }

    //event削除
    if (ev) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    cudaGetLastError();   // 失敗の後始末で出たエラーを次の CUDA 呼び出しへ持ち越さない
    cudaSetDevice(g_openglDeviceID);
}

//コーデックコンテキスト初期化
void save_encode::initialized_ffmpeg_codec_context(int i,int max_split){
    //エンコーダの取得とコンテキスト作成
    const AVCodec* codec = avcodec_find_encoder_by_name(encodeSettings.codec.toUtf8().constData());
    if (!codec) throw_text(QStringLiteral("エンコーダ %1 がありません").arg(encodeSettings.codec));

    ve[i]->codec_ctx = avcodec_alloc_context3(codec);
    if (!ve[i]->codec_ctx) throw_text(QStringLiteral("メモリ不足（エンコーダ）"));

    //これは codec_ctx->pix_fmt に設定するものです
    enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
    for (int k = 0; ; k++) {                     // ★ 引数 i のシャドーイングを解消
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, k);
        if (!config) {
            fprintf(stderr, "Encoder %s does not support any hardware config.\n", codec->name);
            break;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX ) {
            hw_pix_fmt = config->pix_fmt; // この config->pix_fmt が NVENC が実際に期待するフォーマットです
            break;
        }
    }
    if (hw_pix_fmt == AV_PIX_FMT_NONE) {
        fprintf(stderr, "No suitable hardware pixel format found for encoder %s.\n", codec->name);
    }

    //メタデータ設定
    ve[i]->codec_ctx->width = width_/encodeSettings.width_tile;
    ve[i]->codec_ctx->height = height_/encodeSettings.height_tile;
    ve[i]->codec_ctx->pix_fmt = hw_pix_fmt;
    ve[i]->codec_ctx->sw_pix_fmt = out10 ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;
    // 8bit / 10bit とも BT.709 limited で変換している（encode_8bit.cu / encode_10bit.cu）。その旨を書いておく
    //   HDR は非対応なので、原色・伝達特性も BT.709（SDR）で固定
    ve[i]->codec_ctx->colorspace      = AVCOL_SPC_BT709;
    ve[i]->codec_ctx->color_range     = AVCOL_RANGE_MPEG;
    ve[i]->codec_ctx->color_primaries = AVCOL_PRI_BT709;
    ve[i]->codec_ctx->color_trc       = AVCOL_TRC_BT709;

    //フレームレート設定
    AVRational fps = av_d2q(encodeSettings.save_fps, 100000);
    ve[i]->codec_ctx->framerate = fps;
    ve[i]->codec_ctx->time_base = av_inv_q(fps);

    //初期化された hw_* を参照
    ve[i]->codec_ctx->hw_device_ctx = av_buffer_ref(ve[i]->hw_device_ctx);
    ve[i]->codec_ctx->hw_frames_ctx = av_buffer_ref(ve[i]->hw_frames_ctx);
    if (!ve[i]->codec_ctx->hw_device_ctx || !ve[i]->codec_ctx->hw_frames_ctx)
        throw_text(QStringLiteral("メモリ不足（エンコーダ）"));

    // ★ 自分が書き込むファイルの flags を見ること（別ファイル構成では
    //   fmt_ctx はタイル 0 のものなので、そちらを見ると i>0 で食い違う）
    AVFormatContext* const own = ve[i]->fmt;
    if (own && (own->oformat->flags & AVFMT_GLOBALHEADER))
        ve[i]->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    //ビットレート周りの設定
    AVDictionary* opts = nullptr;

    // ★ "rc" は必ず明示する。省略すると FFmpeg が bit_rate / global_quality /
    //   tune から推測するので、版によって選ばれ方が変わる。
    //   ビットレートは 1 ストリームあたり（docs/encode-settings.md）。
    const int quality = qBound(1, encodeSettings.cq, 51);

    if (encodeSettings.rc_mode == "cqp") {
        // --- 固定量子化 ---
        //   ビットレートには一切触らない。品質は常に一定で、容量は素材次第。
        av_dict_set(&opts, "rc", "constqp", 0);
        av_dict_set_int(&opts, "qp", quality, 0);
    } else if (encodeSettings.rc_mode == "cq") {
        // --- CQ（品質指定VBR）---
        //   ★ bit_rate は 0 のままにすること（= -b:v 0）。
        //     目標ビットレートを与えると NVENC はそちらを優先し、
        //     cq は上限としてしか効かなくなる。
        //   最大ビットレートは「ここまでは出してよい」という上限として使う。
        av_dict_set(&opts, "rc", "vbr", 0);
        av_dict_set_int(&opts, "cq", quality, 0);
        ve[i]->codec_ctx->bit_rate = 0;
        if (encodeSettings.max_bit_rate > 0) {
            ve[i]->codec_ctx->rc_max_rate    = encodeSettings.max_bit_rate;
            ve[i]->codec_ctx->rc_buffer_size = encodeSettings.max_bit_rate;
        }
    } else if (encodeSettings.rc_mode == "vbr") {
        // --- VBR ---
        av_dict_set(&opts, "rc", "vbr", 0);
        ve[i]->codec_ctx->bit_rate       = encodeSettings.target_bit_rate;
        ve[i]->codec_ctx->rc_max_rate    = encodeSettings.max_bit_rate;
        ve[i]->codec_ctx->rc_buffer_size = encodeSettings.max_bit_rate;
    } else {
        // --- CBR ---
        //   bufsize = 目標にすると 1 秒窓で張り付く。
        av_dict_set(&opts, "rc", "cbr", 0);
        ve[i]->codec_ctx->bit_rate       = encodeSettings.target_bit_rate;
        ve[i]->codec_ctx->rc_max_rate    = encodeSettings.max_bit_rate;
        ve[i]->codec_ctx->rc_buffer_size = encodeSettings.target_bit_rate;
    }

    // 共通オプション
    av_dict_set(&opts, "preset", encodeSettings.preset.toUtf8().constData(), 0);
    av_dict_set(&opts, "tune", encodeSettings.tune.toUtf8().constData(), 0);
    av_dict_set_int(&opts, "g", encodeSettings.gop_size, 0);
    // ★ ring / async_depth を決めるのに使ったクランプ後の値を渡す。
    //   UI 値をそのまま渡すと、段数計算と実際のリオーダ量がズレる。
    av_dict_set_int(&opts, "bf", b_frames_, 0);
    av_dict_set(&opts, "rc-lookahead", "0", 0);
    // ★ zerolatency は「リオーダなし＝出力を溜めない」ことを要求するチューニング
    //   なので B-frame とは両立しない。b_frames > 0 のときは無効にする。
    av_dict_set(&opts, "zerolatency", b_frames_ > 0 ? "0" : "1", 0);

    // ★ async_depth を上げて NVENC 内部でもパイプラインさせる
    //    （元は "1"。ring との整合はコンストラクタでクランプ済み）
    av_dict_set_int(&opts, "async_depth", async_depth_, 0);

    // ★ NVENC の入力サーフェス数を明示的に固定する。
    //   ffmpeg は nb_surfaces = FFMAX(内部計算値, この指定値) とするので、
    //   こちらから大きめに与えることで「NVENC が最大何枚掴むか」が確定し、
    //   encodeRingSize > nvenc_surfaces_ の不変条件が成立する。
    //   （これを指定しないと版によって内部計算値が ring を超え、
    //     B-frame 使用時に wait_slot_free で止まる）
    av_dict_set_int(&opts, "surfaces", nvenc_surfaces_, 0);

    if(encodeSettings.split_encode_mode=="0"){
        av_dict_set_int(&opts, "split_encode_mode", 0, 0);
    }else{
        av_dict_set_int(&opts, "split_encode_mode", 3, 0);
    }

    // 1pass / 2pass 切り替え
    qDebug()<<encodeSettings.pass_mode;
    if (encodeSettings.pass_mode == "2pass-quarter-res") {
        av_dict_set_int(&opts, "multipass", 1, 0);
    } else if (encodeSettings.pass_mode == "2pass-full-res") {
        av_dict_set_int(&opts, "multipass", 2, 0);
        qDebug()<<"full";
    } else {
        av_dict_set_int(&opts, "multipass", 0, 0);
    }

    // 10bit 出力: HEVC は Main10 を明示（AV1 は入力が P010 なら 10bit になる）
    if (out10 && encodeSettings.codec.contains("hevc"))
        av_dict_set(&opts, "profile", "main10", 0);

    int ret = avcodec_open2(ve[i]->codec_ctx, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0)
        // NVENC は VRAM 不足も同時使用数の上限（GeForce）も ENOMEM で返す
        throw_text(ret == AVERROR(ENOMEM)
                       ? QStringLiteral("GPU %1: エンコーダを開けません（VRAM 不足か同時使用数の上限）").arg(ve[i]->dev)
                       : QStringLiteral("GPU %1: エンコーダを開けません（%2）").arg(ve[i]->dev).arg(av_err_text(ret)));
}

//CUDAデバイスコンテキスト初期化
void save_encode::initialized_ffmpeg_hardware_context(int i)
{
    int ret = 0;

    // ---- hw_frames_ctx ----
    ve[i]->hw_frames_ctx = av_hwframe_ctx_alloc(ve[i]->hw_device_ctx);
    if (!ve[i]->hw_frames_ctx) {
        throw_text(QStringLiteral("メモリ不足（フレーム）"));
    }

    AVHWFramesContext* frames_ctx =
        (AVHWFramesContext*)(ve[i]->hw_frames_ctx->data);

    frames_ctx->format    = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = out10 ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;   // 10bit 出力は P010
    frames_ctx->width     = width_  / encodeSettings.width_tile;
    frames_ctx->height    = height_ / encodeSettings.height_tile;
    frames_ctx->initial_pool_size = encodeRingSize + 4;
    ret = av_hwframe_ctx_init(ve[i]->hw_frames_ctx);
    if (ret < 0)
        throw_text(QStringLiteral("GPU %1: VRAM 不足（エンコード用フレーム。%2）")
                       .arg(ve[i]->dev).arg(av_err_text(ret)));

    // hw_frameリング確保
    ve[i]->hw_frames.resize(encodeRingSize);

    // リングバッファ構築 AVFrame/cudamalloc
    for (int j = 0; j < encodeRingSize; j++) {
        AVFrame* f = av_frame_alloc();
        if (!f) throw_text(QStringLiteral("メモリ不足（フレーム）"));

        f->format = AV_PIX_FMT_CUDA;
        f->width  = frames_ctx->width;
        f->height = frames_ctx->height;

        ret = av_hwframe_get_buffer(ve[i]->hw_frames_ctx, f, 0);
        if (ret < 0) {
            av_frame_free(&f);
            throw_text(QStringLiteral("GPU %1: VRAM 不足（エンコード用フレーム。%2）")
                           .arg(ve[i]->dev).arg(av_err_text(ret)));
        }
        ve[i]->hw_frames[j].frame = f;
    }
}

//オーディオエンコーダー初期化
void save_encode::init_audio_encoder()
{
    // ★ UI は入らない組み合わせを出さないが、ini は手で書き換えられる。
    const QString want = encodeSettings.audio_codec.isEmpty()
                             ? QStringLiteral("aac") : encodeSettings.audio_codec;

    const AVCodec* codec = avcodec_find_encoder_by_name(want.toUtf8().constData());
    if (!codec) {
        qWarning() << "[save_encode] 音声コーデック" << want << "が見つかりません。aac にします";
        codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    }
    if (!codec) throw_text(QStringLiteral("音声エンコーダがありません"));

    const QString muxer = fmt_ctx->oformat->name;
    if (codec->id == AV_CODEC_ID_FLAC && muxer.contains("mov")) {
        qWarning() << "[save_encode] MOV は FLAC を格納できません。aac にします";
        codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!codec) throw_text(QStringLiteral("音声エンコーダがありません"));
    }

    // ★ 可逆はビットレートを指定できない。PCM はサンプル幅で、FLAC は
    //   中身で決まる。立てても無視されるだけだが、立てない方が
    //   「UI でビットレートを無効にする条件」と一対一で読める。
    const bool lossless = (qstrncmp(codec->name, "pcm_", 4) == 0) ||
                          (codec->id == AV_CODEC_ID_FLAC);

    audio_enc_ctx = avcodec_alloc_context3(codec);
    if (!audio_enc_ctx) throw_text(QStringLiteral("メモリ不足（音声エンコーダ）"));
    audio_enc_ctx->sample_rate = VideoInfo.in_sample_rate;
    audio_enc_ctx->time_base = AVRational{1, audio_enc_ctx->sample_rate};
    if (!lossless) audio_enc_ctx->bit_rate = encodeSettings.audio_bitrate;

    av_channel_layout_default(
        &audio_enc_ctx->ch_layout,
        VideoInfo.audio_channels > 0 ? VideoInfo.audio_channels : 2
        );

    // ★ AVCodec::sample_fmts は FFmpeg 7.x で非推奨・新しい版では削除なので
    //   触らずに決め打つ（対応表は docs/encode-settings.md）。
    switch (codec->id) {
    case AV_CODEC_ID_FLAC:      audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_S16;  break;
    case AV_CODEC_ID_PCM_S16LE: audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_S16;  break;
    case AV_CODEC_ID_PCM_S24LE: audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_S32;  break;
    case AV_CODEC_ID_PCM_F32LE: audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLT;  break;
    default:                    audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP; break;
    }

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(audio_enc_ctx, codec, nullptr) < 0)
        throw_text(QStringLiteral("音声エンコーダを開けません"));

    // ★ PCM は frame_size を持たない。0 のまま FIFO の条件に使うと無限ループ。
    audio_frame_size_ = (audio_enc_ctx->frame_size > 0)
                            ? audio_enc_ctx->frame_size : 1024;

    qDebug() << "[save_encode] 音声 =" << codec->name
             << " fmt =" << av_get_sample_fmt_name(audio_enc_ctx->sample_fmt)
             << " ch =" << audio_enc_ctx->ch_layout.nb_channels
             << " rate =" << audio_enc_ctx->sample_rate
             << " frame =" << audio_frame_size_
             << " gain =" << encodeSettings.audio_gain_db << "dB"
             << (lossless ? " (可逆)" : "");

    audio_stream = avformat_new_stream(fmt_ctx, nullptr);
    if (!audio_stream)
        throw_text(QStringLiteral("メモリ不足（音声ストリーム）"));

    audio_stream->time_base = audio_enc_ctx->time_base;

    if (avcodec_parameters_from_context(audio_stream->codecpar, audio_enc_ctx) < 0)
        throw_text(QStringLiteral("音声の設定を書けません"));

    audio_fifo = av_audio_fifo_alloc(
        audio_enc_ctx->sample_fmt,
        audio_enc_ctx->ch_layout.nb_channels,
        8192
        );
    if (!audio_fifo)
        throw_text(QStringLiteral("メモリ不足（音声）"));

    AVSampleFormat in_fmt = VideoInfo.out_format;
    if (in_fmt == AV_SAMPLE_FMT_NONE)
        in_fmt = AV_SAMPLE_FMT_S16;

    int ret = swr_alloc_set_opts2(
        &swr_enc,
        &audio_enc_ctx->ch_layout,
        audio_enc_ctx->sample_fmt,
        audio_enc_ctx->sample_rate,
        &audio_enc_ctx->ch_layout,
        in_fmt,
        audio_enc_ctx->sample_rate,
        0,
        nullptr
        );

    if (ret < 0 || !swr_enc)
        throw_text(QStringLiteral("音声の変換を準備できません"));

    if (swr_init(swr_enc) < 0)
        throw_text(QStringLiteral("音声の変換を準備できません"));

    audio_pts = 0;
}

//エンコード
void save_encode::encode(VideoFrame Frame, cudaTextureObject_t srcTex,
                         cudaEvent_t glWaitEvent, cudaEvent_t* convertDone)
{
    if (!Frame.audio_pcm.isEmpty() && audio_on_)
    {
        AudioJob job;

        job.audio_pcm =
            Frame.audio_pcm;

        {
            std::lock_guard<std::mutex>
                lock(audioMutex);

            audioQueue.push(
                std::move(job)
                );
        }
        audioCV.notify_one();
    }

    encode_video(Frame, srcTex, glWaitEvent, convertDone);
}

//映像エンコード（メインスレッド側 = 変換と発行のみ。CPUはブロックしない）
void save_encode::encode_video(VideoFrame Frame, cudaTextureObject_t srcTex,
                               cudaEvent_t glWaitEvent, cudaEvent_t* convertDone)
{
    if (convertDone) *convertDone = nullptr;

    if (srcTex == 0) {
        qWarning() << "[save_encode] encode_video: srcTex is null";
        return;
    }
    // 途中で失敗していたら何もしない（呼び出し側が failed() を見て終える）
    if (failed_) return;

    const int slot = encodeRingNo;

    // ==========================================================
    // 0) ★ 上書き防止バリア
    //    この slot を使っていた「前のフレーム」が全エンコーダで
    //    完全に終わる（= ワーカーが送出し、NVENC が入力refを離す）まで待つ。
    //    ここを通過した時点で hw_frames[slot] は誰も参照していない。
    //    ★ slot を指定して待つ。出力順に依存しない判定になっている。
    // ==========================================================
    for (auto& e : ve) {
        wait_slot_free(*e, slot);
    }

    // ==========================================================
    // 1) primary GPUの場合、hw_frameを直接書き込み先にする
    // ==========================================================
    for (int i = 0; i < (int)ve.size(); i++)
    {
        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID)
        {
            AVFrame* out = ve[i]->hw_frames[slot].frame;
            ve[i]->hw_frames[slot].d_y      = out->data[0];
            ve[i]->hw_frames[slot].y_pitch  = out->linesize[0];
            ve[i]->hw_frames[slot].d_uv     = out->data[1];
            ve[i]->hw_frames[slot].uv_pitch = out->linesize[1];
        }
    }

    // ==========================================================
    // 1.5) ★ GLの描画完了を待つ（GPU側で待つ。CPUは止めない）
    //    呼び出し側が map 直後に interop_stream へ記録したイベント。
    //    これを踏まないと、シェーダがFBOにラスタライズしている最中に
    //    変換カーネルが読み始めてティアリングする。
    // ==========================================================
    if (glWaitEvent) {
        ck(cudaStreamWaitEvent(st, glWaitEvent, 0), "描画の完了待ち");
    }

    // ==========================================================
    // 2) CUDA NV12変換
    //    行ごとのカーネル 1 本（encode_rows.cuh。8bit / 10bit、タイル数によらない）。
    //    圧縮転送のタイルは XferCodec の詰めた並びへ直接書く（前処理 xfer_fwd を兼ねる。encodeRowsTiles）
    //    テクスチャは元の大きさ（texW_ / texH_）で渡す。書くのはタイル（width_ / height_ の範囲）だけ
    // ==========================================================
    {
        const NV12Tiles t = encodeRowsTiles(slot);
        bool ok;
        if (!srcRgba16)
            ok = CUDA_IMG_Proc->encode_rows_rgba8_to_nv12(srcTex, t, texW_, texH_, srcFlip, st);
        else if (out10)
            ok = CUDA_IMG_Proc->encode_rows_rgba16_to_p010(srcTex, t, texW_, texH_, srcFlip, st);
        else
            ok = CUDA_IMG_Proc->encode_rows_rgba16_to_nv12(srcTex, t, texW_, texH_, srcFlip, encodeSettings.dither, st);   // 8bit 化のディザは設定
        if (!ok) fail(QStringLiteral("変換カーネルのエラー（%1）").arg(QString::fromUtf8(cudaGetErrorString(cudaGetLastError()))));
    }

    // ==========================================================
    // 3) ★ 変換完了を「イベントで記録するだけ」。CPUは待たない。
    //
    //    以前はここで cudaEventSynchronize していたが、その理由は
    //      ・texture object を破棄する前に完了させる
    //      ・別ストリーム(ve[i]->st)のコピーに変換結果を見せる
    //    の2つだった。前者は呼び出し側のキャッシュ化で不要になり、
    //    後者は下の cudaStreamWaitEvent で GPU 側に移した。
    //
    //    これで convert(N+1) が copy(N) / encode(N) と本当に重なる。
    // ==========================================================
    ck(cudaEventRecord(ev, st), "変換の完了の記録");

    // 呼び出し側は、このイベントを interop_stream に待たせてから
    // FBO を unmap すること（変換中にGLが上書きするのを防ぐ）
    if (convertDone) *convertDone = ev;

    // ==========================================================
    // 4) 各GPUへ転送を「発行するだけ」（待たない）
    //    ★ どの経路も、まず変換完了(ev)を GPU 側で待つ。
    //      cudaStreamWaitEvent はデバイス跨ぎで有効。
    //    表示用 GPU のタイル: 変換結果が NVENC の入力フレームに入っているので ready を記録するだけ
    //    別 GPU のタイル    : transfer_remote（メインメモリ経由。圧縮あり / なし）。準備に失敗していれば transfer_peer
    // ==========================================================
    for (int i = 0; i < (int)ve.size(); i++)
    {
        FrameSlot& fs = ve[i]->hw_frames[slot];

        // 変換が終わるまでこのストリームを進ませない
        ck(cudaStreamWaitEvent(ve[i]->st, ev, 0), "変換の完了待ち");

        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID)
        {
            // 変換結果が直接 hw_frame に入っているのでコピー不要。
            // 直前の wait があるので、この ready は変換完了後に発火する。
            ck(cudaEventRecord(fs.ready, ve[i]->st), "フレームの完了の記録");
            continue;
        }

        if (ve[i]->remoteXfer)
            transfer_remote(*ve[i], fs, xferStage_);
        else
            transfer_peer(*ve[i], fs);
    }

    // ==========================================================
    // 5) 各エンコーダスレッドへ投入（ブロックしない）
    //    以降の「コピー完了待ち → send_frame → drain → mux」は
    //    エンコーダごとのワーカーが並列に行う。
    // ==========================================================
    for (int i = 0; i < (int)ve.size(); i++)
    {
        submit_job(*ve[i], slot, frame_index);
    }

    encodeRingNo++;
    if (encodeRingNo >= encodeRingSize)
        encodeRingNo = 0;
    xferStage_ = (xferStage_ + 1) % kXferStage;
    log_xfer_stats();

    frame_index++;
}

// ==============================================================
// 1 タイルの大きさ（書き出す大きさ width_ / height_ をタイル数で割ったもの。P010 は 1 要素 2 バイト）
// ==============================================================
save_encode::TileSize save_encode::tileSize() const
{
    const int w = width_ / encodeSettings.width_tile;
    const int h = height_ / encodeSettings.height_tile;
    const size_t bps = out10 ? 2 : 1;
    return { w, h, size_t(w) * bps, size_t((w + 1) & ~1) * bps };
}

// ==============================================================
// 行ごとの変換カーネル（encode_rows.cuh）に渡すタイル情報
//   普段: hw_frames[slot] の d_y / d_uv（表示用 GPU のタイルは NVENC の入力フレーム、別 GPU のタイルは転送の中継）
//   圧縮転送のタイル: XferCodec の次の組の詰めた並び（packed）。その組を前回使い終えるまで変換を待たせる
// ==============================================================
NV12Tiles save_encode::encodeRowsTiles(int slot)
{
    NV12Tiles t = {};
    t.count = (int)ve.size();
    t.cols  = (t.count == 1) ? 1 : encodeSettings.width_tile;
    t.outW  = (t.count == 1) ? width_  : width_  / encodeSettings.width_tile;
    t.outH  = (t.count == 1) ? height_ : height_ / encodeSettings.height_tile;
    for (int i = 0; i < t.count && i < 8; i++) {
        VideoEncoder& e = *ve[i];
        if (e.codec) {
            cudaEvent_t reuse = nullptr;
            uint8_t* tight = e.codec->nextTight(&reuse);
            if (reuse) ck(cudaStreamWaitEvent(st, reuse, 0), "圧縮転送の作業領域の空き待ち");
            t.y  [i] = tight + e.codec->planeOffset(0);
            t.yp [i] = e.codec->planeRowBytes(0);
            t.uv [i] = tight + e.codec->planeOffset(1);
            t.uvp[i] = e.codec->planeRowBytes(1);
            t.packed[i] = 1;
            t.delta = e.codec->delta() ? 1 : 0;   // 全タイル同じ設定（xfer_delta）
        } else {
            const FrameSlot& fs = e.hw_frames[slot];
            t.y  [i] = fs.d_y;
            t.yp [i] = fs.y_pitch;
            t.uv [i] = fs.d_uv;
            t.uvp[i] = fs.uv_pitch;
        }
    }
    return t;
}

// ==============================================================
// 別 GPU への転送（メインメモリ経由。VideoEncoder::remoteXfer。docs/ENCODE_CORE.md 6）
//   圧縮なし（2 段コピー）
//     表示用 GPU e.st : (変換完了 ev 待ち。呼び出し側) → (h2d_ev[s] 待ち) → D2H → h_y/h_uv[s] → d2h_ev[s]
//     別 GPU   e.rst  : (d2h_ev[s] 待ち) → H2D → AVFrame → h2d_ev[s]、fs.ready
//   圧縮あり（XferCodec。デコードの圧縮転送と同じもので、向きが逆）
//     表示用 GPU（codec の組のストリーム）: (ev 待ち) → 圧縮 → (h2d_ev[s] 待ち) → メインメモリの枠 s
//       前処理（差分・詰めた並び）は変換カーネルが済ませている（encodeRowsTiles → compressPacked）
//     別 GPU   e.rst : (compressedEvent(s) 待ち) → 取り込み・展開・差分戻し → AVFrame → h2d_ev[s]、fs.ready
//   PCIe は上り下りが独立なので、表示用 GPU の書き出しと別 GPU の読み込みは重なる。
//   枠 s の再利用は前回の読み込み完了（h2d_ev[s]）を待つ。AVFrame と中継の再利用は wait_slot_free が守る
// ==============================================================
bool save_encode::init_remote_xfer(VideoEncoder& e)
{
    const TileSize ts = tileSize();
    bool ok = true;

    e.d2h_ev.assign(kXferStage, nullptr);
    e.h2d_ev.assign(kXferStage, nullptr);
    for (int s = 0; s < kXferStage && ok; s++)
        ok = cudaEventCreateWithFlags(&e.d2h_ev[s], cudaEventDisableTiming) == cudaSuccess;
    if (ok) {
        cudaSetDevice(e.dev);
        ok = cudaStreamCreateWithFlags(&e.rst, cudaStreamNonBlocking) == cudaSuccess;
        for (int s = 0; s < kXferStage && ok; s++)
            ok = cudaEventCreateWithFlags(&e.h2d_ev[s], cudaEventDisableTiming) == cudaSuccess;
        cudaSetDevice(g_openglDeviceID);
    }

    // 圧縮（設定 xfer_codec）。初期化できなければ圧縮なし
    const XferCodec::Algo algo = XferCodec::parse(encodeSettings.xfer_codec);
    if (ok && algo != XferCodec::None) {
        e.codec = std::make_unique<XferCodec>();
        if (!e.codec->init(g_openglDeviceID, e.dev, algo, encodeSettings.xfer_delta, out10 ? 2 : 1,
                           { ts.w }, { ts.h }, kXferStage)) {
            qWarning() << "[save_encode] GPU" << e.dev << "圧縮転送の準備に失敗したので圧縮なし";
            e.codec.reset();
        }
        cudaSetDevice(g_openglDeviceID);
    }

    // 圧縮なしの中継（ページ固定・Portable）
    if (ok && !e.codec) {
        e.h_y.assign(kXferStage, nullptr);
        e.h_uv.assign(kXferStage, nullptr);
        for (int s = 0; s < kXferStage && ok; s++) {
            ok = cudaHostAlloc((void**)&e.h_y[s],  ts.yBytes  * ts.h,       cudaHostAllocPortable) == cudaSuccess
              && cudaHostAlloc((void**)&e.h_uv[s], ts.uvBytes * (ts.h / 2), cudaHostAllocPortable) == cudaSuccess;
        }
    }
    if (!ok) {
        qWarning() << "[save_encode] GPU" << e.dev << "メインメモリ経由の転送の準備に失敗したのでデバイス間コピーで送る";
        cudaGetLastError();
        release_remote_xfer(e);
        return false;
    }
    if (!e.codec)
        qDebug() << "[save_encode] GPU" << e.dev << "メインメモリ経由の 2 段コピー（"
                 << kXferStage << "枠 ×" << (ts.yBytes * ts.h * 3 / 2) / (1024 * 1024) << "MB）";
    return true;
}

void save_encode::release_remote_xfer(VideoEncoder& e)
{
    e.codec.reset();   // 中で両方の GPU を同期してから解放する
    if (e.rst || !e.h2d_ev.empty()) {
        cudaSetDevice(e.dev);
        if (e.rst) { cudaStreamSynchronize(e.rst); cudaStreamDestroy(e.rst); e.rst = nullptr; }
        for (auto& ev : e.h2d_ev) if (ev) { cudaEventDestroy(ev); ev = nullptr; }
    }
    cudaSetDevice(g_openglDeviceID);
    for (auto& ev : e.d2h_ev) if (ev) { cudaEventDestroy(ev); ev = nullptr; }
    for (auto& p : e.h_y)  if (p) { cudaFreeHost(p); p = nullptr; }
    for (auto& p : e.h_uv) if (p) { cudaFreeHost(p); p = nullptr; }
    e.h_y.clear(); e.h_uv.clear(); e.d2h_ev.clear(); e.h2d_ev.clear();
    e.remoteXfer = false;
}

void save_encode::transfer_remote(VideoEncoder& e, FrameSlot& fs, int s)
{
    const TileSize ts = tileSize();
    AVFrame* out = fs.frame;

    if (e.codec) {
        // 枠 s の前回（3 フレーム前）の計測。終わっていなければ飛ばす（log_xfer_stats）
        XferCodec::Stats r;
        if (e.codec->takeStats(s, r)) {
            xferFrame_.compressMs = (std::max)(xferFrame_.compressMs, r.compressMs);
            xferFrame_.sendMs     = (std::max)(xferFrame_.sendMs, r.sendMs);
            xferFrame_.expandMs   = (std::max)(xferFrame_.expandMs, r.expandMs);
            xferFrame_.rawBytes  += r.rawBytes;
            xferFrame_.compBytes += r.compBytes;
        }
        // 表示用 GPU: 圧縮 → メインメモリの枠 s（変換完了 ev の後。枠 s は前回の読み込み完了の後）
        //   戻り値は起動と CUDA API の失敗。GPU 上の失敗（nvCOMP のチャンク等）は deviceError で数フレーム遅れて分かる
        const bool enc = e.codec->compressPacked(s, ev, e.h2d_ev[s]);
        // 別 GPU: 取り込み・展開・差分戻し → AVFrame
        bool ok = ck(cudaSetDevice(e.dev), "GPU の選択")
               && ck(cudaStreamWaitEvent(e.rst, e.codec->compressedEvent(s), 0), "圧縮の完了待ち");
        const bool dec = ok && enc && e.codec->decompress(s, 0, e.rst,
            { { out->data[0], size_t(out->linesize[0]), out->data[1], size_t(out->linesize[1]) } });
        if (!dec) fail(QStringLiteral("GPU %1 への圧縮転送に失敗（フレーム %2）").arg(e.dev).arg(frame_index));
        if (e.codec->deviceError()) fail(QStringLiteral("GPU %1 への圧縮転送で GPU 上のエラー（nvCOMP）").arg(e.dev));
        // 終わりの印は失敗しても記録する（待つ側が止まらないように）
        ck(cudaEventRecord(e.h2d_ev[s], e.rst), "転送の完了の記録");
        ck(cudaEventRecord(fs.ready, e.rst), "フレームの完了の記録");
        cudaSetDevice(g_openglDeviceID);
        return;
    }

    // 表示用 GPU: D2H（e.st は呼び出し側で変換完了を待たせてある）
    ck(cudaStreamWaitEvent(e.st, e.h2d_ev[s], 0), "転送の枠の空き待ち");
    ck(cudaMemcpy2DAsync(e.h_y[s],  ts.yBytes,  fs.d_y,  fs.y_pitch,  ts.yBytes,  ts.h,     cudaMemcpyDeviceToHost, e.st), "転送（GPU → メインメモリ）");
    ck(cudaMemcpy2DAsync(e.h_uv[s], ts.uvBytes, fs.d_uv, fs.uv_pitch, ts.uvBytes, ts.h / 2, cudaMemcpyDeviceToHost, e.st), "転送（GPU → メインメモリ）");
    ck(cudaEventRecord(e.d2h_ev[s], e.st), "転送の完了の記録");

    // 別 GPU: H2D → AVFrame
    ck(cudaSetDevice(e.dev), "GPU の選択");
    ck(cudaStreamWaitEvent(e.rst, e.d2h_ev[s], 0), "転送の完了待ち");
    ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], e.h_y[s],  ts.yBytes,  ts.yBytes,  ts.h,     cudaMemcpyHostToDevice, e.rst), "転送（メインメモリ → GPU）");
    ck(cudaMemcpy2DAsync(out->data[1], out->linesize[1], e.h_uv[s], ts.uvBytes, ts.uvBytes, ts.h / 2, cudaMemcpyHostToDevice, e.rst), "転送（メインメモリ → GPU）");
    ck(cudaEventRecord(e.h2d_ev[s], e.rst), "転送の完了の記録");
    ck(cudaEventRecord(fs.ready, e.rst), "フレームの完了の記録");
    cudaSetDevice(g_openglDeviceID);
}

// メインメモリ経由の準備に失敗したとき（primary_ctx で開けない等）: デバイスをまたぐコピー
//   P2P が無い環境ではドライバがメインメモリ経由で処理し、遅い（16K で約 2.4GB/s 程度）
void save_encode::transfer_peer(VideoEncoder& e, FrameSlot& fs)
{
    const TileSize ts = tileSize();
    AVFrame* out = fs.frame;
    ck(cudaMemcpy2DAsync(out->data[0], out->linesize[0], fs.d_y,  fs.y_pitch,  ts.yBytes,  ts.h,     cudaMemcpyDeviceToDevice, e.st), "転送（デバイス間）");
    ck(cudaMemcpy2DAsync(out->data[1], out->linesize[1], fs.d_uv, fs.uv_pitch, ts.uvBytes, ts.h / 2, cudaMemcpyDeviceToDevice, e.st), "転送（デバイス間）");
    ck(cudaEventRecord(fs.ready, e.st), "フレームの完了の記録");
}

// 圧縮転送の計測（1 フレーム分 = タイルの最大）を積み、kXferLogFrames フレームごとに平均をログへ
void save_encode::log_xfer_stats()
{
    if (xferFrame_.compBytes <= 0) return;
    xferSum_.compressMs += xferFrame_.compressMs;
    xferSum_.sendMs     += xferFrame_.sendMs;
    xferSum_.expandMs   += xferFrame_.expandMs;
    xferSum_.rawBytes   += xferFrame_.rawBytes;
    xferSum_.compBytes  += xferFrame_.compBytes;
    xferFrame_ = {};
    if (++xferFrames_ < kXferLogFrames) return;
    qDebug().noquote() << QString("[save_encode] 圧縮転送（%1 フレーム平均、タイルの最大）: 圧縮 %2 ms / 送出 %3 ms / 展開 %4 ms、圧縮率 %5")
                              .arg(xferFrames_)
                              .arg(xferSum_.compressMs / xferFrames_, 0, 'f', 2)
                              .arg(xferSum_.sendMs / xferFrames_, 0, 'f', 2)
                              .arg(xferSum_.expandMs / xferFrames_, 0, 'f', 2)
                              .arg(xferSum_.rawBytes / xferSum_.compBytes, 0, 'f', 2);
    xferSum_ = {};
    xferFrames_ = 0;
}

// ==============================================================
// エンコーダスレッド
// ==============================================================
void save_encode::start_encoder_threads()
{
    for (int i = 0; i < (int)ve.size(); i++) {
        ve[i]->th = std::thread(&save_encode::encoder_loop, this, i);
    }
}

void save_encode::stop_encoder_threads()
{
    // キューの末尾に eos を積む（投入済みフレームは全部処理されてから終わる）
    for (auto& e : ve) {
        if (!e->th.joinable()) continue;
        {
            std::lock_guard<std::mutex> lk(e->mtx);
            EncodeJob j;
            j.eos = true;
            e->jobs.push(j);
        }
        e->job_cv.notify_one();
    }

    for (auto& e : ve) {
        if (e->th.joinable()) e->th.join();
    }

    qDebug() << "[save_encode] encoder threads joined";
}

void save_encode::encoder_loop(int idx)
{
    cudaSetDevice(g_openglDeviceID);   // event / stream は全て primary 側に作ってある

    // ==========================================================
    // ★ フォアグラウンド依存を切る
    //    Windows のフォアグラウンドブーストに左右されないよう、
    //    スレッド優先度を明示的に固定し MMCSS に登録する。
    //    （これが無いと別ウィンドウを前面に出したとき fps が落ちる）
    // ==========================================================
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), kEncoderThreadPriority);
    MmcssScope mmcss(L"Pro Audio", AVRT_PRIORITY_HIGH);
#endif

    VideoEncoder& enc = *ve[idx];

    for (;;)
    {
        EncodeJob job;
        {
            std::unique_lock<std::mutex> lk(enc.mtx);
            enc.job_cv.wait(lk, [&]{ return !enc.jobs.empty(); });
            job = enc.jobs.front();
            enc.jobs.pop();
        }

        // ---------------- 終了処理 ----------------
        if (job.eos) {
            int ret = avcodec_send_frame(enc.codec_ctx, nullptr);
            if (ret < 0 && ret != AVERROR_EOF)
                fail(QStringLiteral("GPU %1: エンコーダのエラー（%2）").arg(enc.dev).arg(av_err_text(ret)));
            drain_video_encoder(enc);          // EOF まで吐き切る
            break;
        }

        FrameSlot& fs = enc.hw_frames[job.slot];

        // ---------------- GPUコピー完了待ち ----------------
        // ★ここが各エンコーダ並列。メインスレッドは既に次フレームの変換に入っている。
        const cudaError_t syncErr = cudaEventSynchronize(fs.ready);
        if (syncErr != cudaSuccess)
            fail(QStringLiteral("GPU %1: 処理エラー（%2）").arg(enc.dev).arg(QString::fromUtf8(cudaGetErrorString(syncErr))));

        AVFrame* f = fs.frame;
        f->pts = job.pts;

        // ---------------- NVENC へ投入 ----------------
        bool sent = false;
        int  spin = 0;
        for (;;)
        {
            int ret = avcodec_send_frame(enc.codec_ctx, f);

            if (ret == 0) { sent = true; break; }

            if (ret == AVERROR(EAGAIN)) {
                // NVENC の入力サーフェスが埋まっている → packet を回収して空ける
                drain_video_encoder(enc);
                if (++spin > 64) {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    spin = 0;
                }
                continue;
            }

            qDebug() << "[enc" << idx << "] send_frame error:" << ret;
            fail(QStringLiteral("GPU %1: エンコーダのエラー（%2）").arg(enc.dev).arg(av_err_text(ret)));
            break;
        }

        // ---------------- slot をワーカーの手から離す ----------------
        // ★ 送出の成否にかかわらずここで落とす。
        //   以降その slot を守るのは AVFrame の参照カウントだけ。
        //   送出できなかった場合は ref を取られていないので即座に空きになる。
        {
            std::lock_guard<std::mutex> lk(enc.mtx);
            enc.hw_frames[job.slot].submitted_to_worker = false;
        }
        enc.slot_cv.notify_all();

        if (!sent)
            continue;

        // ---------------- 出せる packet を回収 ----------------
        drain_video_encoder(enc);
    }
}

//映像フレームドレイン（ワーカースレッド内でのみ呼ぶ）
void save_encode::drain_video_encoder(VideoEncoder& enc)
{
    AVPacket* pkt = enc.pkt;

    for (;;) {
        int ret = avcodec_receive_packet(enc.codec_ctx, pkt);

        if (ret == 0) {
            av_packet_rescale_ts(pkt,
                                 enc.codec_ctx->time_base,
                                 enc.stream->time_base);

            pkt->stream_index = enc.stream->index;

            {
                // ★ 自分のファイルへ書く。1 ファイル構成では全タイルが
                //   同じものを指すので、ロックは共通で取る。
                QMutexLocker locker(&muxMutex);
                int wret = av_interleaved_write_frame(enc.fmt, pkt);
                if (wret < 0) {
                    qDebug() << "write_frame error:" << wret << av_err_text(wret);
                    fail(write_error_text(wret));   // ディスクの空き不足など。呼び出し側が止める
                }
            }

            av_packet_unref(pkt);

            // ★ packet が出た = NVENC が入力フレームの参照を離した
            //    → その slot を1つ解放してメインスレッドに通知
            release_slot(enc);
            continue;
        }

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_unref(pkt);
            break;
        }

        qDebug() << "receive_packet error:" << ret;
        fail(QStringLiteral("GPU %1: エンコーダのエラー（%2）").arg(enc.dev).arg(av_err_text(ret)));
        av_packet_unref(pkt);
        break;
    }
}

// ==============================================================
// slot 占有制御（★前フレーム上書き防止の中核）
//
//   この slot が空いている条件は次の2つ。両方満たすまで待つ。
//     1) ワーカーがこの slot を送出済み
//        （= cudaEventSynchronize(ready) を抜けて send_frame を終えた）
//     2) NVENC がこの AVFrame の参照を離した（ref_count == 1）
//
//   旧実装の occupied = submitted - completed は
//     ・出力パケット順 == 入力フレーム順
//     ・1フレーム = 1パケット
//   を仮定しており、B-frame や複数パケットで崩れると
//   「まだ読まれている面」を空きと誤判定して上書きしていた。
//   上の2条件はどちらの仮定も必要としない。
// ==============================================================
void save_encode::wait_slot_free(VideoEncoder& enc, int slot)
{
    std::unique_lock<std::mutex> lk(enc.mtx);

    const auto slot_free = [&]{
        FrameSlot& fs = enc.hw_frames[slot];
        if (fs.submitted_to_worker) return false;
        return frame_is_unreferenced(fs.frame);
    };

    // 取りこぼし通知があっても進まないよう、待ち続けつつ定期的に警告する。
    // （ここで打ち切って進むと画が壊れるので、絶対に抜けない）
    //
    // ★ 詰まったときにどちらの条件で止まっているか分かるようログを出す。
    //     in_worker=true  -> ワーカーが送出前（GPUコピー待ち等）
    //     refcnt   >1     -> NVENC がまだ入力を掴んでいる
    //                        （= ring が hold_max に対して小さい）
    while (!enc.slot_cv.wait_for(lk, std::chrono::seconds(5), slot_free)) {
        const FrameSlot& fs = enc.hw_frames[slot];
        const int refcnt = (fs.frame && fs.frame->buf[0])
                               ? av_buffer_get_ref_count(fs.frame->buf[0]) : -1;

        qWarning() << "[save_encode] wait_slot_free still waiting."
                   << " slot ="      << slot
                   << " in_worker =" << fs.submitted_to_worker
                   << " refcnt ="    << refcnt
                   << " b_frames ="  << b_frames_
                   << " async ="     << async_depth_
                   << " surfaces ="  << nvenc_surfaces_
                   << " ring ="      << enc.ring_capacity
                   << " submitted =" << (qulonglong)enc.submitted
                   << " completed =" << (qulonglong)enc.completed;

        if (enc.completed == 0) {
            qWarning() << "   -> packet が1本も出ていない。原因は2つのどちらか:"
                       << "\n      (1) ring <= NVENC の入力サーフェス数"
                          " -> コンストラクタの nvenc_surfaces_ の式"
                          " (3 * frameIntervalP + 12) の係数/切片を上げる"
                       << "\n      (2) b_frames が NVENC の上限を超えている"
                          " -> nvenc_max_bframes() の表を下げる"
                          " (av1=7 / hevc=5 / h264=4)";
        }
    }
}

void save_encode::submit_job(VideoEncoder& enc, int slot, int64_t pts)
{
    {
        std::lock_guard<std::mutex> lk(enc.mtx);
        EncodeJob j;
        j.slot = slot;
        j.pts  = pts;
        j.eos  = false;
        enc.jobs.push(j);
        enc.hw_frames[slot].submitted_to_worker = true;   // ★
        enc.submitted++;
    }
    enc.job_cv.notify_one();
}

// パケットを1本回収した＝NVENCが入力refを離した可能性がある、の通知
void save_encode::release_slot(VideoEncoder& enc)
{
    {
        std::lock_guard<std::mutex> lk(enc.mtx);
        enc.completed++;                 // 統計用
    }
    enc.slot_cv.notify_all();            // ★ ref_count が下がったので再評価させる
}

//音声エンコード
void save_encode::encode_audio(AudioJob Frame)
{
    if (!audio_enc_ctx || !audio_fifo || !swr_enc || failed_) return;

    const int ch     = audio_enc_ctx->ch_layout.nb_channels;
    // 入力形式はデコード側が決める（現在はインターリーブ float）
    const int in_bps = av_get_bytes_per_sample(VideoInfo.out_format);

    // ★ 0dB のときは配列に触らない。x 1.0f でも丸めが入り、
    //   pcm_f32le へ等倍で出したときのビット一致が崩れる。
    const bool  use_gain  = (std::fabs(encodeSettings.audio_gain_db) > 1e-9);
    const float gain      = use_gain
                              ? (float)std::pow(10.0, encodeSettings.audio_gain_db / 20.0)
                              : 1.0f;
    // float 出力のときは頭打ちしない（0dBFS 超えを保持できるのが取り柄）
    const bool  need_clamp = (audio_enc_ctx->sample_fmt != AV_SAMPLE_FMT_FLT);

    for (const QByteArray& pcm : Frame.audio_pcm) {
        int in_samples = pcm.size() / (ch * in_bps);
        if (in_samples <= 0) continue;

        // ★ float 出力のときは頭打ちしない（0dBFS 超えを保持できるのが取り柄）。
        QByteArray clamped;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(pcm.constData());

        if (VideoInfo.out_format == AV_SAMPLE_FMT_FLT && (use_gain || need_clamp)) {
            clamped = pcm;                       // 暗黙共有。data() で複製される
            float*    q = reinterpret_cast<float*>(clamped.data());
            const int n = clamped.size() / (int)sizeof(float);
            for (int k = 0; k < n; k++) {
                float v = use_gain ? q[k] * gain : q[k];
                if (need_clamp) {
                    if      (v >  1.0f) v =  1.0f;
                    else if (v < -1.0f) v = -1.0f;
                }
                q[k] = v;
            }
            src = reinterpret_cast<const uint8_t*>(clamped.constData());
        }

        const uint8_t* in_data[1] = { src };

        // ★ 正しい delay 計算
        int max_out_samples = av_rescale_rnd(
            swr_get_delay(swr_enc, VideoInfo.in_sample_rate) + in_samples,
            audio_enc_ctx->sample_rate,
            VideoInfo.in_sample_rate,
            AV_ROUND_UP
            );

        uint8_t** converted = nullptr;
        if (av_samples_alloc_array_and_samples(&converted, nullptr, ch, max_out_samples,
                                               audio_enc_ctx->sample_fmt, 0) < 0) {
            fail(QStringLiteral("メモリ不足（音声）"));
            return;
        }

        const int out_samples = swr_convert(swr_enc, converted, max_out_samples, in_data, in_samples);
        if (out_samples < 0)
            fail(QStringLiteral("音声の変換エラー（%1）").arg(av_err_text(out_samples)));
        else if (out_samples > 0 && av_audio_fifo_write(audio_fifo, (void**)converted, out_samples) < out_samples)
            fail(QStringLiteral("メモリ不足（音声）"));

        av_freep(&converted[0]);
        av_freep(&converted);
        if (failed_) return;
    }

    // ---- 1 フレーム単位で吐き出し（PCM は自分で決めた長さ）----
    const int fs = audio_frame_size_;

    while (!failed_ && av_audio_fifo_size(audio_fifo) >= fs) {
        AVFrame* frame = alloc_audio_frame(fs);
        if (!frame) return;

        // ★ PTS は「取り出したサンプル数」基準
        int read_samples = av_audio_fifo_read(audio_fifo,(void**)frame->data,fs);
        if (read_samples < 0) { av_frame_free(&frame); fail(QStringLiteral("音声の読み出しエラー")); return; }
        frame->pts = audio_pts;
        audio_pts += (int)(((double)read_samples)*(VideoInfo.fps/encodeSettings.save_fps));   // ★ 必ず read_samples で進める

        send_audio(frame);
        av_frame_free(&frame);
    }
}

// 音声フレーム（n サンプル、バッファ付き）。失敗したら fail して nullptr
AVFrame* save_encode::alloc_audio_frame(int n)
{
    AVFrame* f = av_frame_alloc();
    if (f) {
        f->nb_samples  = n;
        f->format      = audio_enc_ctx->sample_fmt;
        f->sample_rate = audio_enc_ctx->sample_rate;
        if (av_channel_layout_copy(&f->ch_layout, &audio_enc_ctx->ch_layout) < 0 ||
            av_frame_get_buffer(f, 0) < 0)
            av_frame_free(&f);
    }
    if (!f) fail(QStringLiteral("メモリ不足（音声）"));
    return f;
}

// 音声 1 フレーム（nullptr = 終わりの合図）をエンコーダへ渡し、出たパケットを書く。失敗は fail
void save_encode::send_audio(AVFrame* f)
{
    int ret = avcodec_send_frame(audio_enc_ctx, f);
    if (ret < 0 && ret != AVERROR_EOF) {
        fail(QStringLiteral("音声エンコーダのエラー（%1）").arg(av_err_text(ret)));
        return;
    }
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) { fail(QStringLiteral("メモリ不足（音声）")); return; }
    for (;;) {
        ret = avcodec_receive_packet(audio_enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) { fail(QStringLiteral("音声エンコーダのエラー（%1）").arg(av_err_text(ret))); break; }
        av_packet_rescale_ts(pkt, audio_enc_ctx->time_base, audio_stream->time_base);
        pkt->stream_index = audio_stream->index;
        {
            QMutexLocker locker(&muxMutex);
            const int wret = av_interleaved_write_frame(fmt_ctx, pkt);
            if (wret < 0) fail(write_error_text(wret));
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
}

//音声エンコード終了
void save_encode::stop_audio_thread()
{
    {
        std::lock_guard<std::mutex>
            lock(audioMutex);

        audioRunning = false;
    }

    audioCV.notify_all();

    if (audioThread.joinable())
        audioThread.join();

    qDebug() << "audio joined";
}

//音声エンコードスレッドループ
void save_encode::audio_loop()
{
    // ★ 映像ワーカーの優先度を上げたぶん、音声が相対的に押されないようにする
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), kAudioThreadPriority);
    MmcssScope mmcss(L"Audio", AVRT_PRIORITY_NORMAL);
#endif

    while (audioRunning)
    {
        AudioJob job;

        {
            std::unique_lock<std::mutex> lock(audioMutex);

            audioCV.wait(lock, [&] {
                return !audioQueue.empty()
                || !audioRunning;
            });

            if (!audioRunning &&
                audioQueue.empty())
                break;

            job = std::move(audioQueue.front());
            audioQueue.pop();
        }

        {
            std::lock_guard<std::mutex> lock(audioEncMutex);
            encode_audio(job);
        }
    }

    {
        std::lock_guard<std::mutex> lock(audioEncMutex);
        audio_flush();
    }
}

//音声エンコード終了処理（途中で失敗していたら何もしない。映像側の close が結果を返す）
void save_encode::audio_flush(){
    if (!audio_enc_ctx || !audio_fifo || failed_) return;
    const int fs = audio_frame_size_;
    const int remain = av_audio_fifo_size(audio_fifo);

    if (remain > 0) {
        AVFrame* f = alloc_audio_frame(fs);
        if (!f) return;
        // FIFO → frame、不足分は無音
        if (av_audio_fifo_read(audio_fifo, (void**)f->data, remain) < remain) {
            av_frame_free(&f);
            fail(QStringLiteral("音声の読み出しエラー"));
            return;
        }
        av_samples_set_silence(f->data, remain, fs - remain,
                               audio_enc_ctx->ch_layout.nb_channels, audio_enc_ctx->sample_fmt);
        f->pts = audio_pts;
        audio_pts += remain; // ★ 実サンプル数で進める
        send_audio(f);
        av_frame_free(&f);
    }

    // NULL frame で drain
    send_audio(nullptr);
}
