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


save_encode::save_encode(int h,int w) {
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    // ★ 画像処理インスタンスの生成
    //   これが無いと encode_video() 内の
    //   CUDA_IMG_Proc->rgba_to_nv12xN_split_tex() が nullptr 参照になる。
    //   メンバを触らない実装だと「たまたま動く」ため気づきにくい。
    if (!CUDA_IMG_Proc) {
        CUDA_IMG_Proc = new CUDA_ImageProcess();
    }

    width_=w;
    height_=h;
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
            (double)(width_ / wt) * (double)(height_ / ht) * 1.5 / (1024.0 * 1024.0);
        // ★ 設定画面の「想定使用 VRAM」と同じ式（encode_pool_mb）
        const int total_mb = encode_pool_mb(width_ / wt, height_ / ht,
                                            encodeRingSize, tiles);

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
        if (r < 0 || !f) throw std::runtime_error("Failed to allocate format context");
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
    qDebug()<<encodeSettings.encode_tile;
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
        QString gpuId = QString::number(encodeSettings.tile_gpu_map[i]);
        qDebug()<<gpuId;
        ret = av_hwdevice_ctx_create(&ve[i]->hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
        if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

        initialized_ffmpeg_hardware_context(i);   // ここで hw_frames.resize() される
        initialized_ffmpeg_codec_context(i,encodeSettings.encode_tile);

        // ★ stream 作成だけ
        ve[i]->stream = avformat_new_stream(ve[i]->fmt, nullptr);
        if (!ve[i]->stream) throw std::runtime_error("Failed to create stream");

        ve[i]->stream->time_base = ve[i]->codec_ctx->time_base;
        ve[i]->stream->avg_frame_rate = ve[i]->codec_ctx->framerate;
        ve[i]->stream->r_frame_rate   = ve[i]->codec_ctx->framerate;
        ret = avcodec_parameters_from_context(ve[i]->stream->codecpar, ve[i]->codec_ctx);
        if (ret < 0) throw std::runtime_error("Failed to copy codec parameters");

        //GPU転送用のメモリを確保
        for(int j=0;j<encodeRingSize;j++){
            if(encodeSettings.tile_gpu_map[i] != g_openglDeviceID){
                cudaMallocPitch(
                    &ve[i]->hw_frames[j].d_y,
                    &ve[i]->hw_frames[j].y_pitch,
                    width_/encodeSettings.width_tile,
                    height_/encodeSettings.height_tile
                    );
                cudaMallocPitch(
                    &ve[i]->hw_frames[j].d_uv,
                    &ve[i]->hw_frames[j].uv_pitch,
                    (width_/encodeSettings.width_tile),
                    (height_/encodeSettings.height_tile)/2
                    );
            }
        }

        //Stream作成
        cudaStreamCreateWithFlags(
            &ve[i]->st,
            cudaStreamNonBlocking
            );

        //event作成
        //  cudaEventBlockingSync は効果が無かったので付けない（スピン待ちのまま）
        for(int j=0;j<encodeRingSize;j++){
            cudaEventCreateWithFlags(
                &ve[i]->hw_frames[j].ready,
                cudaEventDisableTiming
                );
        }

        // ★ パイプライン用の初期化
        ve[i]->pkt = av_packet_alloc();
        if (!ve[i]->pkt) throw std::runtime_error("Failed to allocate packet");
        ve[i]->ring_capacity = encodeRingSize;
        ve[i]->submitted = 0;
        ve[i]->completed = 0;
    }

    //音声エンコーダー作成
    //   ★ 「音声を含める」を外しているときは音声ストリーム自体を作らない。
    audio_on_ = (VideoInfo.audio && encodeSettings.audio_enable);
    if(audio_on_){
        init_audio_encoder();
        audioRunning = true;
        audioThread =
            std::thread(
                &save_encode::audio_loop,
                this
                );
    }

    // ③④ 全ファイルを開いてヘッダを書く
    //     ★ 必ず全 stream（映像 N 本 + 音声）を作り終えてから。
    for (int fi = 0; fi < (int)files_.size(); fi++) {
        AVFormatContext* f = files_[fi];
        const QString p = separate_ ? tile_path(fi) : encodeSettings.encode_path;

        ret = avio_open(&f->pb, p.toUtf8().constData(), AVIO_FLAG_WRITE);
        if (ret < 0) throw std::runtime_error("Failed to open output file");

        ret = avformat_write_header(f, nullptr);
        if (ret < 0) throw std::runtime_error("Failed to write header");
    }

    //Stream作成
    cudaStreamCreateWithFlags(
        &st,
        cudaStreamNonBlocking
        );

    //event作成
    cudaEventCreateWithFlags(
        &ev,
        cudaEventDisableTiming
        );

    // ⑤ ★ ヘッダを書き終えてからワーカー起動（mux 可能になってから）
    start_encoder_threads();
}

save_encode::~save_encode() {
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

    // ==========================
    // ③ ファイルの終了処理（mux 参加者が全員止まってから）
    // ==========================
    //   ★ 所有は files_ 1 箇所。fmt_ctx と ve[]->own_fmt は同じものを
    //     指すことがあるので、ここで畳んでから両方 null にする。
    for (AVFormatContext* f : files_) {
        if (!f) continue;
        if (f->pb) avio_flush(f->pb);

        const int tret = av_write_trailer(f);
        qDebug() << "trailer ret =" << tret;

        if (f->pb) avio_closep(&f->pb);
        avformat_free_context(f);
    }
    files_.clear();
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

        for (int j = 0; j < (int)ve[i]->hw_frames.size(); j++) {

            //event削除
            if (ve[i]->hw_frames[j].ready) {
                cudaEventDestroy(ve[i]->hw_frames[j].ready);
                ve[i]->hw_frames[j].ready = nullptr;
            }

            //中間バッファ解放（primary GPU の場合は AVFrame の中身を指しているので触らない）
            if (encodeSettings.tile_gpu_map[i] != g_openglDeviceID) {
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

    qDebug() << "save_encode: Destructor called";
}

//コーデックコンテキスト初期化
void save_encode::initialized_ffmpeg_codec_context(int i,int max_split){
    //エンコーダの取得とコンテキスト作成
    const AVCodec* codec = avcodec_find_encoder_by_name(encodeSettings.codec.toUtf8().constData());
    if (!codec) throw std::runtime_error("hevc_nvenc codec not found");

    ve[i]->codec_ctx = avcodec_alloc_context3(codec);
    if (!ve[i]->codec_ctx) throw std::runtime_error("Failed to allocate AVCodecContext");

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

    //フレームレート設定
    AVRational fps = av_d2q(encodeSettings.save_fps, 100000);
    ve[i]->codec_ctx->framerate = fps;
    ve[i]->codec_ctx->time_base = av_inv_q(fps);

    //初期化された hw_* を参照
    ve[i]->codec_ctx->hw_device_ctx = av_buffer_ref(ve[i]->hw_device_ctx);
    ve[i]->codec_ctx->hw_frames_ctx = av_buffer_ref(ve[i]->hw_frames_ctx);

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

    int ret = avcodec_open2(ve[i]->codec_ctx, codec, &opts);
    if (ret < 0) throw std::runtime_error("Failed to open codec");
    av_dict_free(&opts);
}

//CUDAデバイスコンテキスト初期化
void save_encode::initialized_ffmpeg_hardware_context(int i)
{
    int ret = 0;

    // ---- hw_frames_ctx ----
    ve[i]->hw_frames_ctx = av_hwframe_ctx_alloc(ve[i]->hw_device_ctx);
    if (!ve[i]->hw_frames_ctx) {
        throw std::runtime_error("Failed to allocate hw_frames_ctx");
    }

    AVHWFramesContext* frames_ctx =
        (AVHWFramesContext*)(ve[i]->hw_frames_ctx->data);

    frames_ctx->format    = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width     = width_  / encodeSettings.width_tile;
    frames_ctx->height    = height_ / encodeSettings.height_tile;
    frames_ctx->initial_pool_size = encodeRingSize + 4;
    ret = av_hwframe_ctx_init(ve[i]->hw_frames_ctx);
    if (ret < 0) {
        throw std::runtime_error("Failed to init frames_ctx");
    }

    // hw_frameリング確保
    ve[i]->hw_frames.resize(encodeRingSize);

    // リングバッファ構築 AVFrame/cudamalloc
    for (int j = 0; j < encodeRingSize; j++) {
        AVFrame* f = av_frame_alloc();
        if (!f) throw std::runtime_error("av_frame_alloc failed");

        f->format = AV_PIX_FMT_CUDA;
        f->width  = frames_ctx->width;
        f->height = frames_ctx->height;

        ret = av_hwframe_get_buffer(ve[i]->hw_frames_ctx, f, 0);
        if (ret < 0) {
            av_frame_free(&f);
            throw std::runtime_error("Failed to alloc hw_frame ring buffer");
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
    if (!codec) throw std::runtime_error("audio encoder not found");

    const QString muxer = fmt_ctx->oformat->name;
    if (codec->id == AV_CODEC_ID_FLAC && muxer.contains("mov")) {
        qWarning() << "[save_encode] MOV は FLAC を格納できません。aac にします";
        codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!codec) throw std::runtime_error("audio encoder not found");
    }

    // ★ 可逆はビットレートを指定できない。PCM はサンプル幅で、FLAC は
    //   中身で決まる。立てても無視されるだけだが、立てない方が
    //   「UI でビットレートを無効にする条件」と一対一で読める。
    const bool lossless = (qstrncmp(codec->name, "pcm_", 4) == 0) ||
                          (codec->id == AV_CODEC_ID_FLAC);

    audio_enc_ctx = avcodec_alloc_context3(codec);
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
        throw std::runtime_error("avcodec_open2(audio) failed");

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
        throw std::runtime_error("avformat_new_stream(audio) failed");

    audio_stream->time_base = audio_enc_ctx->time_base;

    if (avcodec_parameters_from_context(audio_stream->codecpar, audio_enc_ctx) < 0)
        throw std::runtime_error("copy audio params failed");

    audio_fifo = av_audio_fifo_alloc(
        audio_enc_ctx->sample_fmt,
        audio_enc_ctx->ch_layout.nb_channels,
        8192
        );
    if (!audio_fifo)
        throw std::runtime_error("av_audio_fifo_alloc failed");

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
        throw std::runtime_error("swr_alloc_set_opts2 failed");

    if (swr_init(swr_enc) < 0)
        throw std::runtime_error("swr_init failed");

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
        cudaStreamWaitEvent(st, glWaitEvent, 0);
    }

    // ==========================================================
    // 2) CUDA NV12変換
    // ==========================================================
    if (ve.size() == 1) {
        CUDA_IMG_Proc->rgba_to_nv12x1_split_tex(
            srcTex,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch,
            ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            width_, height_,
            srcFlip,
            st
            );
    }
    else if (ve.size() == 2) {
        CUDA_IMG_Proc->rgba_to_nv12x2_split_tex(
            srcTex,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch, ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            ve[1]->hw_frames[slot].d_y, ve[1]->hw_frames[slot].y_pitch, ve[1]->hw_frames[slot].d_uv, ve[1]->hw_frames[slot].uv_pitch,
            width_, height_,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            srcFlip,
            st
            );
    }
    else if (ve.size() == 4) {
        CUDA_IMG_Proc->rgba_to_nv12x4_split_tex(
            srcTex,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch, ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            ve[1]->hw_frames[slot].d_y, ve[1]->hw_frames[slot].y_pitch, ve[1]->hw_frames[slot].d_uv, ve[1]->hw_frames[slot].uv_pitch,
            ve[2]->hw_frames[slot].d_y, ve[2]->hw_frames[slot].y_pitch, ve[2]->hw_frames[slot].d_uv, ve[2]->hw_frames[slot].uv_pitch,
            ve[3]->hw_frames[slot].d_y, ve[3]->hw_frames[slot].y_pitch, ve[3]->hw_frames[slot].d_uv, ve[3]->hw_frames[slot].uv_pitch,
            width_, height_,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            srcFlip,
            st
            );
    }
    else if (ve.size() == 8) {
        CUDA_IMG_Proc->rgba_to_nv12x8_split_tex(
            srcTex,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch, ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            ve[1]->hw_frames[slot].d_y, ve[1]->hw_frames[slot].y_pitch, ve[1]->hw_frames[slot].d_uv, ve[1]->hw_frames[slot].uv_pitch,
            ve[2]->hw_frames[slot].d_y, ve[2]->hw_frames[slot].y_pitch, ve[2]->hw_frames[slot].d_uv, ve[2]->hw_frames[slot].uv_pitch,
            ve[3]->hw_frames[slot].d_y, ve[3]->hw_frames[slot].y_pitch, ve[3]->hw_frames[slot].d_uv, ve[3]->hw_frames[slot].uv_pitch,
            ve[4]->hw_frames[slot].d_y, ve[4]->hw_frames[slot].y_pitch, ve[4]->hw_frames[slot].d_uv, ve[4]->hw_frames[slot].uv_pitch,
            ve[5]->hw_frames[slot].d_y, ve[5]->hw_frames[slot].y_pitch, ve[5]->hw_frames[slot].d_uv, ve[5]->hw_frames[slot].uv_pitch,
            ve[6]->hw_frames[slot].d_y, ve[6]->hw_frames[slot].y_pitch, ve[6]->hw_frames[slot].d_uv, ve[6]->hw_frames[slot].uv_pitch,
            ve[7]->hw_frames[slot].d_y, ve[7]->hw_frames[slot].y_pitch, ve[7]->hw_frames[slot].d_uv, ve[7]->hw_frames[slot].uv_pitch,
            width_, height_,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            srcFlip,
            st
            );
    }
    else {
        qWarning() << "[save_encode] unsupported tile count:" << (int)ve.size();
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
    cudaEventRecord(ev, st);

    // 呼び出し側は、このイベントを interop_stream に待たせてから
    // FBO を unmap すること（変換中にGLが上書きするのを防ぐ）
    if (convertDone) *convertDone = ev;

    // ==========================================================
    // 4) 各GPUへ転送を「発行するだけ」（待たない）
    //    ★ どの経路も、まず変換完了(ev)を GPU 側で待つ。
    //      cudaStreamWaitEvent はデバイス跨ぎで有効。
    // ==========================================================
    for (int i = 0; i < (int)ve.size(); i++)
    {
        FrameSlot& fs = ve[i]->hw_frames[slot];

        // 変換が終わるまでこのストリームを進ませない
        cudaStreamWaitEvent(ve[i]->st, ev, 0);

        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID)
        {
            // 変換結果が直接 hw_frame に入っているのでコピー不要。
            // 直前の wait があるので、この ready は変換完了後に発火する。
            cudaEventRecord(fs.ready, ve[i]->st);
            continue;
        }

        AVFrame* out = fs.frame;

        cudaMemcpy2DAsync(
            out->data[0], out->linesize[0],
            fs.d_y, fs.y_pitch,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            cudaMemcpyDeviceToDevice,
            ve[i]->st
            );

        cudaMemcpy2DAsync(
            out->data[1], out->linesize[1],
            fs.d_uv, fs.uv_pitch,
            width_ / encodeSettings.width_tile,
            (height_ / encodeSettings.height_tile) / 2,
            cudaMemcpyDeviceToDevice,
            ve[i]->st
            );

        cudaEventRecord(fs.ready, ve[i]->st);
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

    frame_index++;
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
            if (ret < 0) qDebug() << "[enc" << idx << "] send NULL frame error:" << ret;
            drain_video_encoder(enc);          // EOF まで吐き切る
            break;
        }

        FrameSlot& fs = enc.hw_frames[job.slot];

        // ---------------- GPUコピー完了待ち ----------------
        // ★ここが各エンコーダ並列。メインスレッドは既に次フレームの変換に入っている。
        cudaEventSynchronize(fs.ready);

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
                    char err[256];
                    av_strerror(wret, err, sizeof(err));
                    qDebug() << "write_frame error:" << wret << err;
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
    if (!audio_enc_ctx || !audio_fifo || !swr_enc) return;

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
        av_samples_alloc_array_and_samples(
            &converted,
            nullptr,
            ch,
            max_out_samples,
            audio_enc_ctx->sample_fmt,
            0
            );

        int out_samples = swr_convert(
            swr_enc,
            converted,
            max_out_samples,
            in_data,
            in_samples
            );

        if (out_samples > 0) {
            av_audio_fifo_write(audio_fifo, (void**)converted, out_samples);
        }

        av_freep(&converted[0]);
        av_freep(&converted);
    }

    // ---- 1 フレーム単位で吐き出し（PCM は自分で決めた長さ）----
    const int fs = audio_frame_size_;

    while (av_audio_fifo_size(audio_fifo) >= fs) {
        AVFrame* frame = av_frame_alloc();
        frame->nb_samples = fs;
        frame->format = audio_enc_ctx->sample_fmt;
        frame->sample_rate = audio_enc_ctx->sample_rate;
        av_channel_layout_copy(&frame->ch_layout, &audio_enc_ctx->ch_layout);

        av_frame_get_buffer(frame, 0);

        // ★ PTS は「取り出したサンプル数」基準
        int read_samples = av_audio_fifo_read(audio_fifo,(void**)frame->data,fs);
        frame->pts = audio_pts;
        audio_pts += (int)(((double)read_samples)*(VideoInfo.fps/encodeSettings.save_fps));   // ★ 必ず read_samples で進める

        avcodec_send_frame(audio_enc_ctx, frame);
        av_frame_free(&frame);

        AVPacket* pkt = av_packet_alloc();

        while (avcodec_receive_packet(audio_enc_ctx, pkt) == 0) {
            av_packet_rescale_ts(
                pkt,
                audio_enc_ctx->time_base,
                audio_stream->time_base
                );
            pkt->stream_index = audio_stream->index;

            {
                QMutexLocker locker(&muxMutex);
                av_interleaved_write_frame(fmt_ctx, pkt);
            }

            av_packet_unref(pkt);
        }

        av_packet_free(&pkt);
    }
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

//音声エンコード終了処理
void save_encode::audio_flush(){
    // ==========================
    // Audio flush（AAC）
    // ==========================
    if (audio_enc_ctx && audio_fifo) {
        const int fs = audio_frame_size_;
        int remain = av_audio_fifo_size(audio_fifo);

        if (remain > 0) {
            AVFrame* f = av_frame_alloc();
            f->nb_samples = fs;
            f->format = audio_enc_ctx->sample_fmt;
            f->sample_rate = audio_enc_ctx->sample_rate;
            av_channel_layout_copy(&f->ch_layout, &audio_enc_ctx->ch_layout);

            av_frame_get_buffer(f, 0);

            // FIFO → frame
            av_audio_fifo_read(audio_fifo, (void**)f->data, remain);

            // 不足分を silence で埋める
            av_samples_set_silence(
                f->data,
                remain,
                fs - remain,
                audio_enc_ctx->ch_layout.nb_channels,
                audio_enc_ctx->sample_fmt
                );

            // PTS
            f->pts = audio_pts;
            audio_pts += remain; // ★ 実サンプル数で進める

            avcodec_send_frame(audio_enc_ctx, f);
            av_frame_free(&f);
        }

        // NULL frame で drain
        avcodec_send_frame(audio_enc_ctx, nullptr);

        // ★ packet drain loop
        AVPacket* pkt = av_packet_alloc();

        while (avcodec_receive_packet(audio_enc_ctx, pkt) == 0) {
            av_packet_rescale_ts(
                pkt,
                audio_enc_ctx->time_base,
                audio_stream->time_base
                );

            pkt->stream_index = audio_stream->index;

            {
                QMutexLocker locker(&muxMutex);
                av_interleaved_write_frame(fmt_ctx, pkt);
            }

            av_packet_unref(pkt);
        }

        av_packet_free(&pkt);
    }
}
