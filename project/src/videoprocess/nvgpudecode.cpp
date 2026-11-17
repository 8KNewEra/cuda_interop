#include "src/videoprocess/nvgpudecode.h"
#include "src/videoprocess/decode_gpu_map.h"

#include <QStringList>
#include <QLibrary>
#include <QElapsedTimer>
#include <QVariantMap>
#include <algorithm>

extern "C" {
#include <libavutil/hwcontext_cuda.h>
}

// ===============================
// mp4 NVIDIA向けGPUデコード
// ===============================

// ffmpeg初期化
// ==============================================================
// NVDEC のエンジン数（性能 HUD のデコード上限の目安用）
//
//   1 本のストリーム（タイル）は NVDEC を 1 基しか使わないので、GPU が受け持つタイル数が
//   エンジン数より少ないと残りの基は遊ぶ（例: RTX 5000 Ada は 2 基。1 タイルでは約半分の使用率で頭打ち）。
//   cuvidGetDecoderCaps の nNumNVDECs（Video Codec SDK 12 以降の CUVIDDECODECAPS。以前は予約で 0）で引く。
//   nvcuvid はドライバに付いてくるので、ヘッダ・ライブラリは使わず実行時に読み込む（FFmpeg と同じやり方）。
//   取れなければ 0（HUD は 1 基とみなす）
// ==============================================================
namespace {
struct NvDecCaps {                      // CUVIDDECODECAPS と同じ並び
    int            eCodecType;          // IN  cudaVideoCodec
    int            eChromaFormat;       // IN  cudaVideoChromaFormat
    unsigned int   nBitDepthMinus8;     // IN
    unsigned int   reserved1[3];
    unsigned char  bIsSupported;        // OUT
    unsigned char  nNumNVDECs;          // OUT（SDK 12 以降）
    unsigned short nOutputFormatMask;
    unsigned int   nMaxWidth, nMaxHeight, nMaxMBCount;
    unsigned short nMinWidth, nMinHeight;
    unsigned char  bIsHistogramSupported, nCounterBitDepth;
    unsigned short nMaxHistogramBins;
    unsigned int   reserved3[10];
};
using GetDecoderCapsFn = int (*)(NvDecCaps*);   // CUresult CUDAAPI（x64 は呼び出し規約が 1 つ）

int nvdecEngines(int gpu, AVCodecID codec, int bitdepth)
{
    static GetDecoderCapsFn fn = [] {
#ifdef _WIN32
        static QLibrary lib("nvcuvid");
#else
        static QLibrary lib("nvcuvid", 1);
#endif
        return lib.load() ? reinterpret_cast<GetDecoderCapsFn>(lib.resolve("cuvidGetDecoderCaps")) : nullptr;
    }();
    if (!fn) return 0;
    int type;   // cudaVideoCodec
    switch (codec) {
    case AV_CODEC_ID_H264: type = 4;  break;
    case AV_CODEC_ID_HEVC: type = 8;  break;
    case AV_CODEC_ID_VP9:  type = 10; break;
    case AV_CODEC_ID_AV1:  type = 11; break;
    default: return 0;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(gpu);
    cudaFree(nullptr);   // その GPU の primary context を current にする（cuvid はドライバ API の current を使う）
    NvDecCaps c{};
    c.eCodecType = type;
    c.eChromaFormat = 1;   // 4:2:0
    c.nBitDepthMinus8 = unsigned(std::max(0, bitdepth - 8));
    const int r = fn(&c);
    cudaSetDevice(prev);
    return (r == 0 && c.bIsSupported) ? int(c.nNumNVDECs) : 0;
}
} // namespace

bool nvgpudecode::initialized_ffmpeg()
{
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    packet = av_packet_alloc();
    int ret;

    // ------------------------
    // CUDA デバイスコンテキスト作成
    // ------------------------
    QString gpuId = QString::number(g_openglDeviceID);
    qDebug() << "GPU No:" << gpuId;

    // ==========================================================
    // ★ ffmpeg を我々と同じ CUDA コンテキストに置く
    //
    //   既定(flags=0)だと hwcontext_cuda は cuCtxCreate で独立した
    //   コンテキストを作る。そこに属するストリームは Runtime API から
    //   扱えず、NVDEC の転送完了を待つ手段が無くなる。
    //   ★ flags 引数に AV_CUDA_USE_PRIMARY_CONTEXT を渡すと環境により
    //     -129(ENOTSUP) になる。必ず dict オプションで渡すこと。
    // ==========================================================
    cudaFree(0);                    // プライマリコンテキストを先に作らせる

    bool ctx_shared = false;        // ffmpeg が我々と同じコンテキストにいるか

    {
        // ① primary_ctx=1
        AVDictionary* devopts = nullptr;
        av_dict_set(&devopts, "primary_ctx", "1", 0);
        ret = av_hwdevice_ctx_create(
            &hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
            gpuId.toUtf8().data(), devopts, 0);
        av_dict_free(&devopts);
        ctx_shared = (ret >= 0);

        // ② 駄目なら current_ctx=1（今カレントなプライマリをそのまま使わせる）
        if (ret < 0) {
            qWarning() << "[nvgpudecode] primary_ctx 失敗:" << ffmpegErrStr(ret);
            if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);

            AVDictionary* o2 = nullptr;
            av_dict_set(&o2, "current_ctx", "1", 0);
            ret = av_hwdevice_ctx_create(
                &hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                gpuId.toUtf8().data(), o2, 0);
            av_dict_free(&o2);
            ctx_shared = (ret >= 0);
        }

        // ③ それも駄目なら既定（別コンテキスト）。同期できないので後で警告。
        if (ret < 0) {
            qWarning() << "[nvgpudecode] current_ctx 失敗:" << ffmpegErrStr(ret);
            if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);

            ret = av_hwdevice_ctx_create(
                &hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                gpuId.toUtf8().data(), nullptr, 0);
            ctx_shared = false;
        }
    }

    if (ret < 0) {
        Error_String = QString("av_hwdevice_ctx_create failed (GPU %1): %2")
        .arg(gpuId)
            .arg(ffmpegErrStr(ret));
        return false;
    }

    // ------------------------
    // ファイルを開く
    // ------------------------
    ret = avformat_open_input(&fmt_ctx, input_filename, nullptr, nullptr);
    if (ret < 0) {
        Error_String = QString("avformat_open_input failed: %1")
        .arg(ffmpegErrStr(ret));
        return false;
    }

    // ------------------------
    // ストリーム情報取得
    // ------------------------
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        Error_String = QString("avformat_find_stream_info failed: %1")
        .arg(ffmpegErrStr(ret));
        return false;
    }

    // ------------------------
    // 映像ストリーム検索
    // ------------------------
    std::vector<int> video_stream_indices;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_indices.push_back(i);
        }
    }

    if (video_stream_indices.empty()) {
        Error_String = "No video streams found";
        return false;
    }

    // ------------------------
    // ビット深度取得
    // ------------------------
    AVCodecParameters* par = fmt_ctx->streams[video_stream_indices[0]]->codecpar;
    if (par->codec_id == AV_CODEC_ID_HEVC ||
        par->codec_id == AV_CODEC_ID_H264 ||
        par->codec_id == AV_CODEC_ID_AV1) {

        if (par->format == AV_PIX_FMT_YUV420P10 ||
            par->format == AV_PIX_FMT_P010) {
            VideoInfo.bitdepth = 10;
        } else {
            VideoInfo.bitdepth = 8;
        }
    }

    qDebug() << "bit depth:" << VideoInfo.bitdepth;

    // マルチ GPU デコードの圧縮転送（試験的。nvCOMP）。開くときの設定を使う
    xferAlgo_  = XferCodec::parse(g_AppSettings.decode_xfer_codec);
    xferDelta_ = g_AppSettings.decode_xfer_delta;

    // ------------------------
    // タイルごとのデコード GPU（マルチ GPU デコード。decode_gpu_map.h）
    //   設定（GPU ごとの重み）で按分。既定は全部を表示用 GPU
    // ------------------------
    const std::vector<int> tileGpu = assignDecodeTiles(
        int(video_stream_indices.size()),
        avcodec_get_name(fmt_ctx->streams[video_stream_indices[0]]->codecpar->codec_id));
    VideoInfo.tile_gpu.clear();

    // ------------------------
    // デコーダ作成（GPU）
    // ------------------------
    for (int i = 0; i < video_stream_indices.size(); i++) {
        int stream_index = video_stream_indices[i];

        vd.emplace_back();
        auto& dec = vd.back();
        dec.gpu = (i < (int)tileGpu.size()) ? tileGpu[i] : g_openglDeviceID;

        //リングバッファフレーム作成
        // ★ ゼロコピー化後は、このリングが「デコード結果の保護」そのものになる。
        //   GUIスレッドが hw_frames[j] を直接読むので、一周する前に
        //   上書きされないだけの面数が必要（in-flight は 3〜4 程度）。
        dec.hw_frames.resize(ringSize);
        for (int j = 0; j < ringSize; j++) {
            dec.hw_frames[j] = av_frame_alloc();
            if (!dec.hw_frames[j]) {
                Error_String = "av_frame_alloc failed";
                return false;
            }
        }

        // ★ 受信用スクラッチ（receive_into_slot が使う）
        dec.tmp_frame = av_frame_alloc();
        if (!dec.tmp_frame) {
            Error_String = "av_frame_alloc(tmp) failed";
            return false;
        }

        dec.stream_index = stream_index;
        const AVStream* st = fmt_ctx->streams[stream_index];
        const char* codec_name = avcodec_get_name(st->codecpar->codec_id);

        const char* decoder_name = selectDecoder(codec_name);
        if (!decoder_name) {
            Error_String = QString("Unsupported codec: %1")
            .arg(codec_name);
            return false;
        }

        dec.decoder = avcodec_find_decoder_by_name(decoder_name);
        if (!dec.decoder) {
            Error_String = QString("Decoder not found: %1")
            .arg(decoder_name);
            return false;
        }

        dec.codec_ctx = avcodec_alloc_context3(dec.decoder);
        if (!dec.codec_ctx) {
            Error_String = "avcodec_alloc_context3 failed";
            return false;
        }

        // 表示用 GPU 以外のタイルはその GPU のデバイスコンテキストで開く（NVDEC もその GPU で動く）
        AVBufferRef* devctx = hw_device_ctx;
        if (dec.gpu != g_openglDeviceID) {
            devctx = remote_hw_ctx(dec.gpu);
            if (!devctx) {
                qWarning() << "[nvgpudecode] GPU" << dec.gpu << "で開けないので表示用 GPU でデコードします tile" << i;
                dec.gpu = g_openglDeviceID;
                devctx = hw_device_ctx;
            }
        }
        VideoInfo.tile_gpu.push_back(dec.gpu);
        dec.codec_ctx->hw_device_ctx = av_buffer_ref(devctx);
        dec.codec_ctx->pkt_timebase = st->time_base;

        ret = avcodec_parameters_to_context(dec.codec_ctx, st->codecpar);
        if (ret < 0) {
            Error_String = QString("avcodec_parameters_to_context failed: %1")
            .arg(ffmpegErrStr(ret));
            return false;
        }

        AVDictionary* opts = nullptr;
        // ★ extra_hw_frames は「指定しない」。FFmpeg 既定の -1（自動）に任せる。
        //
        //   *_cuvid は hwaccel ではなく独立したデコーダで、出力サーフェスは
        //   自前のプール（surfaces オプション、既定 25 枚）から確保する。
        //   ここに extra_hw_frames を 1 以上で与えると
        //   ulNumDecodeSurfaces との整合が崩れ、デコード自体は通るのに
        //   シーク（avcodec_flush_buffers + av_seek_frame）後に
        //   "cuvid decode callback error" で落ちる。
        //   0 を明示するのも「追加ゼロ」という明示指示になるので外す。
        //
        //   アプリが hw_frames を ringSize(=6) 枚保持しても、
        //   既定 25 枚のプールに対しては十分余裕がある。
        //   どうしても枚数を明示したい場合は extra_hw_frames ではなく
        //   cuvid の surfaces を使うこと（extra_hw_frames とは併用しない）:
        //     av_dict_set_int(&opts, "surfaces", ringSize + 20, 0);

        ret = avcodec_open2(dec.codec_ctx, dec.decoder, &opts);
        av_dict_free(&opts);

        if (ret < 0) {
            Error_String = QString("avcodec_open2 failed (%1): %2")
            .arg(decoder_name)
                .arg(ffmpegErrStr(ret));
            return false;
        }
    }

    // ------------------------
    // 動画情報取得
    // ------------------------
    VideoInfo.fps = getFrameRate(fmt_ctx, vd[0].stream_index);

    double time_base_d = av_q2d(fmt_ctx->streams[vd[0].stream_index]->time_base);
    VideoInfo.pts_per_frame = 1.0 / (VideoInfo.fps * time_base_d);
    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    // ★ 欠番の多いファイルでは avg_frame_rate が下振れする。実測で補正する。
    fix_frame_grid(vd[0].stream_index, time_base_d);

    if (!get_last_frame_pts()) {
        Error_String = "get_last_frame_pts failed";
        return false;
    }

    // ------------------------
    // デコードモード設定
    // ------------------------
    if (vd.size() == 2) {
        VideoInfo.decode_mode = "Tiled Multi-Stream:stream_x2(tile:2×1)\n";
        VideoInfo.width_scale = 2;
        VideoInfo.height_scale = 1;
    } else if (vd.size() == 4) {
        VideoInfo.decode_mode = "Tiled Multi-Stream:stream_x4(tile:2×2)\n";
        VideoInfo.width_scale = 2;
        VideoInfo.height_scale = 2;
    } else if (vd.size() == 8) {
        VideoInfo.decode_mode = "Tiled Multi-Stream:stream_x8(tile:4×2)\n";
        VideoInfo.width_scale = 4;
        VideoInfo.height_scale = 2;
    } else {
        VideoInfo.decode_mode = "Tiled Multi-Stream:stream_x1(tile:1)\n";
        VideoInfo.width_scale = 1;
        VideoInfo.height_scale = 1;
    }

    // 複数の GPU でデコードするときは割り振りを書き足す（例: "Decode GPU: 0×3 + 1×1"）
    {
        std::map<int, int> per;
        for (int g : VideoInfo.tile_gpu) per[g]++;
        // 各 GPU の NVDEC の基数（性能 HUD の目安用。取れた GPU だけ）
        VideoInfo.gpu_nvdecs.clear();
        const AVCodecParameters* vpar = fmt_ctx->streams[video_stream_indices[0]]->codecpar;
        for (const auto& [g, n] : per) {
            const int e = nvdecEngines(g, vpar->codec_id, VideoInfo.bitdepth);
            if (e > 0) VideoInfo.gpu_nvdecs[g] = e;
            qDebug().noquote() << QString("[nvgpudecode] GPU%1: NVDEC %2 基、%3 タイル").arg(g).arg(e > 0 ? QString::number(e) : "?").arg(n);
        }
        if (per.size() > 1) {
            QStringList parts;
            parts << QString("%1×%2").arg(g_openglDeviceID).arg(per[g_openglDeviceID]);
            for (const auto& [g, n] : per) if (g != g_openglDeviceID) parts << QString("%1×%2").arg(g).arg(n);
            VideoInfo.decode_mode = VideoInfo.decode_mode.trimmed() + "\nDecode GPU: " + parts.join(" + ") + "\n";
            qDebug().noquote() << "[nvgpudecode] マルチ GPU デコード:" << parts.join(" + ");
        }
    }

    // ------------------------
    // CUDA メモリ確保
    // ------------------------
    // ★ ゼロコピー化により、中間の RGBA バッファは入口・出口とも不要。
    //   入口: NV12→RGBA結合はGUIスレッドがOpenGLテクスチャへ直接書き込む
    //   出口: NV12変換カーネルが FBO を texture object 経由で直読みする
    //   16K RGBA 1面 = 約531MB。入口4面 + 出口1面 = 約2.6GB の削減。

    // ==========================================================
    // ★ NVDEC 転送完了イベントの用意
    //
    //   全デコーダが hw_device_ctx を共有しているので、タイルの出力
    //   コピーは全部まとめて同じストリームに積まれる。よって全タイルを
    //   受け取った後に 1 回記録するだけで全タイルの完了を表せる。
    // ==========================================================
    {
        AVHWDeviceContext*   dctx = (AVHWDeviceContext*)hw_device_ctx->data;
        AVCUDADeviceContext* cu   = (AVCUDADeviceContext*)dctx->hwctx;

        // ★ hwcontext_cuda.c は hwctx->stream に *常に NULL* を入れる
        //   （= 既定ストリームを使う、の意味）。正常値なので弾かないこと。
        //   cuvid も NULL ストリームへコピーを積むので対象が一致する。
        dec_stream = (cudaStream_t)cu->stream;

        cudaEvent_t probe = nullptr;
        bool rec_ok = false;
        if (cudaEventCreateWithFlags(&probe, cudaEventDisableTiming) == cudaSuccess) {
            rec_ok = (cudaEventRecord(probe, dec_stream) == cudaSuccess);
            cudaGetLastError();
            cudaEventDestroy(probe);
        }

        dec_sync_by_event = (ctx_shared && rec_ok);

        if (dec_sync_by_event) {
            decode_ready_ev.assign(ringSize, nullptr);
            for (int j = 0; j < ringSize; j++) {
                if (cudaEventCreateWithFlags(&decode_ready_ev[j],
                                             cudaEventDisableTiming) != cudaSuccess) {
                    Error_String = "cudaEventCreate(decode_ready) failed";
                    return false;
                }
            }
            qDebug() << "[nvgpudecode] NVDEC 完了待ち: イベント方式";

        } else {
            decode_ready_ev.clear();

            // ======================================================
            // ★ ここに来たら NVDEC の転送完了を待てていない。
            //   症状はタイルの途中に水平の継ぎ目が出る形の破綻で、
            //   pts は正しいので TILE PTS MISMATCH には引っかからない。
            //
            //  【ctx_shared = false】上に primary_ctx / current_ctx の
            //   失敗行が出ているはず。ffmpeg がオプション名を変えた。
            //   libavutil/hwcontext_cuda.c の cuda_flags_from_opts を見て
            //   現在の名前に合わせる（従来は primary_ctx / current_ctx）。
            //
            //  【rec_ok = false】ffmpeg のエラーは無いのに記録できない。
            //   hwctx->stream が Runtime API から触れない値になった。
            //   cuda_context_init を見る（従来は常に NULL = 既定ストリーム）。
            //
            //  【警告は出ないのに割れる】イベントの記録先と cuvid の
            //   コピー先がズレている。Driver API の NULL は常にレガシー
            //   既定ストリームだが、Runtime API の 0 は
            //   --default-stream per-thread 付きでビルドするとスレッド毎の
            //   既定ストリームになる。その構成にするなら dec_stream へ
            //   cudaStreamLegacy を明示すること。
            //   また interop_stream は cudaStreamCreate（blocking）のままに
            //   しておくこと。レガシー既定ストリームとの暗黙同期が保険になる。
            //
            //  【コンテキストをどうしても共有できない場合】
            //   確実に待つには Driver API を使う（cuda.h / cuda.lib が必要、
            //   CPU は止まる）:
            //       CUcontext prev;
            //       cuCtxPushCurrent(cu->cuda_ctx);
            //       cuStreamSynchronize(cu->stream);
            //       cuCtxPopCurrent(&prev);
            //   ※ publish_decode_frame のフォールバックにある
            //      cudaDeviceSynchronize() はカレントコンテキストしか
            //      待たないので、この状況では役に立たない。
            //
            //  【再現手順】16K タイル動画の再生中に裏で YouTube を流す。
            //   NVDEC を取り合わせるのが最も出やすい。
            // ======================================================
            qWarning() << "[nvgpudecode] NVDEC 完了待ちを確立できません"
                       << "ctx_shared =" << ctx_shared << "rec_ok =" << rec_ok
                       << "- 映像が割れる可能性があります";
        }
    }

    // ------------------------
    // 音声ストリーム探索
    // ------------------------
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    if (audio_stream_index == -1) {
        Error_String = "Audio stream not found";
        VideoInfo.audio = false;
    } else {
        audio_frame = av_frame_alloc();

        audio_decoder =
            avcodec_find_decoder(fmt_ctx->streams[audio_stream_index]->codecpar->codec_id);
        audio_ctx = avcodec_alloc_context3(audio_decoder);

        avcodec_parameters_to_context(
            audio_ctx,
            fmt_ctx->streams[audio_stream_index]->codecpar
            );

        ret = avcodec_open2(audio_ctx, audio_decoder, nullptr);
        if (ret < 0) {
            Error_String = QString("avcodec_open2(audio) failed: %1")
            .arg(ffmpegErrStr(ret));
            audio_stream_index = -1;
            return false;
        }

        VideoInfo.in_sample_rate = audio_ctx->sample_rate;
        VideoInfo.in_format      = audio_ctx->sample_fmt;
        av_channel_layout_copy(&in_ch_layout, &audio_ctx->ch_layout);

        VideoInfo.out_sample_rate = audio_ctx->sample_rate;
        // ★ float のまま配る。S16 へ先に落とすと 0dBFS 超えがここで潰れ、
        //   音量を下げても歪みが残る（audio_decode::apply_volume 参照）。
        VideoInfo.out_format      = AV_SAMPLE_FMT_FLT;
        av_channel_layout_default(&out_ch_layout, 2);

        // ★ チャンネル数はここで入れること。
        //   以前は get_decode_audio() の中（= 最初の音声パケットを変換した
        //   時点）でしか代入しておらず、しかもそれはエンコード中しか走らない。
        //   save_encode の init_audio_encoder() はエンコーダ構築時に
        //   VideoInfo.audio_channels を読むので、初期値の 1 を拾って
        //   モノラルで開いていた。レイアウトが決まるこの場所が正しい。
        VideoInfo.audio_channels  = out_ch_layout.nb_channels;

        if (swr) {
            swr_free(&swr);
        }

        ret = swr_alloc_set_opts2(
            &swr,
            &out_ch_layout,
            VideoInfo.out_format,
            VideoInfo.out_sample_rate,
            &in_ch_layout,
            VideoInfo.in_format,
            VideoInfo.in_sample_rate,
            0,
            nullptr
            );

        if (ret < 0 || swr_init(swr) < 0) {
            Error_String = "swr_init failed";
            return false;
        }


        VideoInfo.audio = true;
    }

    // ------------------------
    // 再生初期化
    // ------------------------
    emit send_slider_info();

    video_play_flag = true;
    video_reverse_flag = false;
    slider_No = 0;

    return true;
}


//デコーダ設定
const char*nvgpudecode::selectDecoder(const char* codec_name) {
    const char*codec{};
    if (strcmp(codec_name, "h264") == 0) {
        codec="h264_cuvid";
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc_cuvid";
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1_cuvid";
    }else{
        codec=nullptr;
    }
    // ★ nullptr を std::string へ代入してはいけない（未定義動作）。
    //   対応していないコーデックのときは表示名を変えずに返す。
    //   呼び出し側は戻り値の null を見てエラーにする。
    if (codec) VideoInfo.Codec = codec;
    return codec;
}

//映像取得
void nvgpudecode::get_decode_image(){
    if(vd.size()==0){
        return;
    }else if(vd.size()==1){
        get_singledecode_image();
    }else{
        get_multidecode_image();
    }
}


// ==============================================================
// ★ デストラクタ：保留パケットを解放する
// ==============================================================
nvgpudecode::~nvgpudecode()
{
    clear_pending();
    release_remote();
}

// ==============================================================
// ★ マルチ GPU デコード（decode_gpu_map.h）
//
//   表示用 GPU 以外の GPU でデコードしたタイルは、その GPU のメモリにある。GUI のカーネルは
//   表示用 GPU で動くので、publish_decode_frame で表示用 GPU 上のリング（vd[i].r_y / r_uv）へ転送する。
//   ・転送は「その GPU → メインメモリ（ページ固定）→ 表示用 GPU」の 2 段（transfer_remote）。
//     ドライバ任せの GPU 間コピー（P2P が無い環境）は途中で止まりやすく、帯域が出なかった（16K 4 タイルで約 2.4GB/s）
//   ・1 段目は cuvid の出力コピーと同じ、その GPU の既定ストリーム（NULL）に積むので、デコード結果の完成後に走る
//   ・2 段目と decode_ready は表示用 GPU の非ブロッキングストリーム（既定ストリームを止めない）
//     GUI 側は従来どおり decode_ready だけを待てばよい
// ==============================================================
AVBufferRef* nvgpudecode::remote_hw_ctx(int gpu)
{
    auto it = remote_ctx_.find(gpu);
    if (it != remote_ctx_.end()) return it->second;

    AVBufferRef* ctx = nullptr;
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "primary_ctx", "1", 0);   // Runtime API（イベント・転送）と同じコンテキストに置く
    const int ret = av_hwdevice_ctx_create(&ctx, AV_HWDEVICE_TYPE_CUDA,
                                           QString::number(gpu).toUtf8().constData(), opts, 0);
    av_dict_free(&opts);
    if (ret < 0) {
        qWarning() << "[nvgpudecode] GPU" << gpu << "のデバイスコンテキストを作れません:" << ffmpegErrStr(ret);
        return nullptr;
    }
    remote_ctx_[gpu] = ctx;
    return ctx;
}

bool nvgpudecode::init_remote()
{
    if (remote_init_) return true;
    cudaSetDevice(g_openglDeviceID);
    if (cudaStreamCreateWithFlags(&xfer_stream_, cudaStreamNonBlocking) != cudaSuccess) return false;
    for (int s = 0; s < kStage; s++) {
        cudaEventCreateWithFlags(&local_ev_[s], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&h2d_ev_[s],   cudaEventDisableTiming);
    }
    for (auto& [g, ctx] : remote_ctx_) {
        Q_UNUSED(ctx);
        cudaSetDevice(g);
        RemoteGpu& r = remote_[g];
        for (int s = 0; s < kStage; s++) {   // 時間を測るので DisableTiming は付けない
            cudaEventCreate(&r.beg[s]);
            cudaEventCreate(&r.end[s]);
        }
    }
    cudaSetDevice(g_openglDeviceID);
    remote_init_ = (cudaGetLastError() == cudaSuccess);
    return remote_init_;
}

bool nvgpudecode::alloc_remote_tile(int i, AVFrame* f)
{
    VideoDecorder& d = vd[i];
    if (!d.r_y.empty()) return true;
    const int    bps    = VideoInfo.bitdepth > 8 ? 2 : 1;   // P010 は 1 サンプル 2 バイト
    const size_t wBytes = size_t(f->width) * bps;
    const int    h      = f->height;

    cudaSetDevice(g_openglDeviceID);
    // 転送先（表示用 GPU。hw_frames と同じ段数。GUI はこれを読む）
    d.r_y.assign(ringSize, nullptr);
    d.r_uv.assign(ringSize, nullptr);
    for (int j = 0; j < ringSize; j++) {
        if (cudaMallocPitch((void**)&d.r_y[j],  &d.r_ypitch,  wBytes, h)     != cudaSuccess ||
            cudaMallocPitch((void**)&d.r_uv[j], &d.r_uvpitch, wBytes, h / 2) != cudaSuccess) {
            cudaGetLastError();
            qWarning() << "[nvgpudecode] 転送先の確保に失敗 tile" << i;
            return false;
        }
    }
    // 中継（メインメモリのページ固定。Portable = どの GPU のコンテキストからも非同期コピーできる）
    d.h_y.assign(kStage, nullptr);
    d.h_uv.assign(kStage, nullptr);
    for (int s = 0; s < kStage; s++) {
        if (cudaHostAlloc((void**)&d.h_y[s],  wBytes * h,       cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**)&d.h_uv[s], wBytes * (h / 2), cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            qWarning() << "[nvgpudecode] 中継バッファ（ページ固定メモリ）の確保に失敗 tile" << i;
            return false;
        }
    }
    d.h_rowBytes = wBytes;
    d.h_rows     = h;
    qDebug().noquote() << QString("[nvgpudecode] tile%1: GPU%2 → メインメモリ → GPU%3 で転送（1 枚 %4 MB）")
                              .arg(i).arg(d.gpu).arg(g_openglDeviceID).arg(wBytes * h * 1.5 / 1e6, 0, 'f', 1);
    return true;
}

// 別 GPU のタイルを表示用 GPU のリング（r_y / r_uv の ringNo）へ送り、decode_ready を記録する
bool nvgpudecode::transfer_remote(const std::vector<int>& tiles)
{
    if (!init_remote()) { qWarning() << "[nvgpudecode] 転送の準備に失敗"; return false; }
    const int s = stage_;
    stage_ = (stage_ + 1) % kStage;

    // 圧縮転送（設定が「なし」以外）。準備に失敗したらこのファイルの間は圧縮しない（下の 2 段コピーへ）
    if (xferAlgo_ != XferCodec::None) {
        if (transfer_compressed(tiles, s)) return true;
        if (xferAlgo_ != XferCodec::None) return false;   // 圧縮の途中で失敗（このコマは前フレームを再送）
    }

    // 1 段目: その GPU → メインメモリ（GPU ごとにまとめて、時間も測る）
    std::vector<int> gpus;
    for (int i : tiles) if (std::find(gpus.begin(), gpus.end(), vd[i].gpu) == gpus.end()) gpus.push_back(vd[i].gpu);
    double doneMs = 0, doneBytes = 0;
    bool ok = true;
    for (int g : gpus) {
        RemoteGpu& r = remote_[g];
        cudaSetDevice(g);
        // 前回この枠を使ったときの D2H の時間（もう終わっていれば。待たない）
        if (r.used[s] && cudaEventQuery(r.end[s]) == cudaSuccess) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, r.beg[s], r.end[s]) == cudaSuccess) { doneMs += ms; doneBytes += r.bytes[s]; }
        }
        cudaGetLastError();
        // 中継の枠が空くまで（前回の H2D の完了まで）待つ
        if (h2d_used_[s]) cudaStreamWaitEvent(0, h2d_ev_[s], 0);
        cudaEventRecord(r.beg[s], 0);
        double bytes = 0;
        for (int i : tiles) {
            if (vd[i].gpu != g) continue;
            const VideoDecorder& d = vd[i];
            const AVFrame* f = d.hw_frames[ringNo];
            ok = ok
              && cudaMemcpy2DAsync(d.h_y[s],  d.h_rowBytes, f->data[0], size_t(f->linesize[0]),
                                   d.h_rowBytes, d.h_rows,     cudaMemcpyDeviceToHost, 0) == cudaSuccess
              && cudaMemcpy2DAsync(d.h_uv[s], d.h_rowBytes, f->data[1], size_t(f->linesize[1]),
                                   d.h_rowBytes, d.h_rows / 2, cudaMemcpyDeviceToHost, 0) == cudaSuccess;
            bytes += double(d.h_rowBytes) * d.h_rows * 1.5;
        }
        cudaEventRecord(r.end[s], 0);
        r.used[s]  = true;
        r.bytes[s] = bytes;
    }
    cudaSetDevice(g_openglDeviceID);
    if (doneMs > 0) emit transfer_time_ms(doneMs, doneBytes / (doneMs * 1e6));   // 性能 HUD（ms と GB/s）

    // 2 段目: メインメモリ → 表示用 GPU（xfer_stream_。表示用 GPU のタイルの完了と D2H の完了を待ってから）
    cudaEventRecord(local_ev_[s], dec_stream);
    cudaStreamWaitEvent(xfer_stream_, local_ev_[s], 0);
    for (int g : gpus) cudaStreamWaitEvent(xfer_stream_, remote_[g].end[s], 0);
    for (int i : tiles) {
        const VideoDecorder& d = vd[i];
        ok = ok
          && cudaMemcpy2DAsync(d.r_y[ringNo],  d.r_ypitch,  d.h_y[s],  d.h_rowBytes,
                               d.h_rowBytes, d.h_rows,     cudaMemcpyHostToDevice, xfer_stream_) == cudaSuccess
          && cudaMemcpy2DAsync(d.r_uv[ringNo], d.r_uvpitch, d.h_uv[s], d.h_rowBytes,
                               d.h_rowBytes, d.h_rows / 2, cudaMemcpyHostToDevice, xfer_stream_) == cudaSuccess;
    }
    cudaEventRecord(h2d_ev_[s], xfer_stream_);
    h2d_used_[s] = true;
    if (!ok) {
        qWarning() << "[nvgpudecode] タイルの転送に失敗:" << cudaGetErrorString(cudaGetLastError());
        return false;
    }

    // decode_ready（GUI が待つ）= このコマの全タイルの完成
    if (dec_sync_by_event && ringNo < (int)decode_ready_ev.size() && decode_ready_ev[ringNo]
        && cudaEventRecord(decode_ready_ev[ringNo], xfer_stream_) == cudaSuccess) {
        Frame.decode_ready = decode_ready_ev[ringNo];
    } else {
        cudaGetLastError();
        cudaStreamSynchronize(xfer_stream_);
        Frame.decode_ready = nullptr;
    }
    return true;
}

// ==============================================================
// ★ 圧縮転送（nvCOMP。xfer_codec.h）
//   デコード側 GPU で圧縮 → メインメモリ（実際のバイト数だけ）→ 表示用 GPU で展開して r_y / r_uv の ringNo へ。
//   順序の取り方は 2 段コピーと同じ（枠 s の再利用は h2d_ev_、表示用 GPU のタイルは local_ev_、完了は decode_ready）
// ==============================================================
bool nvgpudecode::transfer_compressed(const std::vector<int>& tiles, int s)
{
    // GPU ごとにまとめる（XferCodec は 1 枚の GPU が受け持つタイルをまとめて扱う）
    std::map<int, std::vector<int>> byGpu;
    for (int i : tiles) byGpu[vd[i].gpu].push_back(i);

    const int bps = VideoInfo.bitdepth > 8 ? 2 : 1;
    // 時間は GPU ごとに並行して進むので最大、バイト数は合計
    double cmpMs = 0, sendMs = 0, decMs = 0, raw = 0, comp = 0;
    double fetchMs = 0, decompMs = 0, invMs = 0;   // 展開の内訳（表示用 GPU の 1 本のストリームで順に進むので合計）
    bool haveStats = false, ok = true;
    for (auto& [g, idx] : byGpu) {
        std::unique_ptr<XferCodec>& c = codecs_[g];
        if (!c) {
            std::vector<int> w, h;
            for (int i : idx) { w.push_back(vd[i].hw_frames[ringNo]->width); h.push_back(vd[i].hw_frames[ringNo]->height); }
            c = std::make_unique<XferCodec>();
            if (!c->init(g, g_openglDeviceID, xferAlgo_, xferDelta_, bps, w, h, kStage)) {
                qWarning() << "[nvgpudecode] 圧縮転送を使えないので、圧縮しない転送に戻します";
                codecs_.clear();
                cudaSetDevice(g_openglDeviceID);
                xferAlgo_ = XferCodec::None;
                return false;
            }
        }
        // この枠の前回の結果（圧縮率と時間。性能 HUD 用。待たない）
        double cm, sm, d, rb, cb, fm = -1, xm = -1, im = -1;
        if (c->takeStats(s, cm, sm, d, rb, cb, &fm, &xm, &im)) {
            cmpMs = (std::max)(cmpMs, cm); sendMs = (std::max)(sendMs, sm); decMs += d;   // 展開は表示用 GPU の 1 本のストリームで順に
            fetchMs += (std::max)(0.0, fm); decompMs += (std::max)(0.0, xm); invMs += (std::max)(0.0, im);
            raw += rb; comp += cb; haveStats = true;
        }
        std::vector<const AVFrame*> frames;
        for (int i : idx) frames.push_back(vd[i].hw_frames[ringNo]);
        ok = ok && c->encode(s, frames, h2d_used_[s] ? h2d_ev_[s] : nullptr);
    }
    cudaSetDevice(g_openglDeviceID);
    if (haveStats && cmpMs + sendMs > 0) {
        // 実効速度 = 元の大きさ ÷ 圧縮と送るの遅い方（フレームごとに交互の組で重ねて進むので、合計ではなく遅い方で流れる）
        emit transfer_time_ms(sendMs, raw / ((std::max)(cmpMs, sendMs) * 1e6));
        emit transfer_codec_stats(comp > 0 ? raw / comp : 0.0, cmpMs, decMs);  // 圧縮率、圧縮と展開の時間
        emit transfer_expand_parts(fetchMs, decompMs, invMs);                // 展開の内訳
    }

    // 表示用 GPU: 表示用 GPU のタイルの完了と、各 GPU の圧縮の完了を待ってから展開
    cudaEventRecord(local_ev_[s], dec_stream);
    cudaStreamWaitEvent(xfer_stream_, local_ev_[s], 0);
    for (auto& [g, idx] : byGpu) cudaStreamWaitEvent(xfer_stream_, codecs_[g]->encodedEvent(s), 0);
    for (auto& [g, idx] : byGpu) {
        std::vector<XferCodec::Dst> dst;
        for (int i : idx) dst.push_back({ vd[i].r_y[ringNo], vd[i].r_ypitch, vd[i].r_uv[ringNo], vd[i].r_uvpitch });
        ok = ok && codecs_[g]->decode(s, xfer_stream_, dst);
    }
    cudaSetDevice(g_openglDeviceID);
    cudaEventRecord(h2d_ev_[s], xfer_stream_);
    h2d_used_[s] = true;
    if (!ok) return false;

    if (dec_sync_by_event && ringNo < (int)decode_ready_ev.size() && decode_ready_ev[ringNo]
        && cudaEventRecord(decode_ready_ev[ringNo], xfer_stream_) == cudaSuccess) {
        Frame.decode_ready = decode_ready_ev[ringNo];
    } else {
        cudaGetLastError();
        cudaStreamSynchronize(xfer_stream_);
        Frame.decode_ready = nullptr;
    }
    return true;
}

void nvgpudecode::release_remote()
{
    codecs_.clear();   // 圧縮転送のバッファ（中で各 GPU を待ってから解放する）
    for (auto& [g, r] : remote_) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
        for (int s = 0; s < kStage; s++) {
            if (r.beg[s]) cudaEventDestroy(r.beg[s]);
            if (r.end[s]) cudaEventDestroy(r.end[s]);
        }
    }
    remote_.clear();
    cudaSetDevice(g_openglDeviceID);
    if (xfer_stream_) { cudaStreamSynchronize(xfer_stream_); cudaStreamDestroy(xfer_stream_); xfer_stream_ = nullptr; }
    for (int s = 0; s < kStage; s++) {
        if (local_ev_[s]) { cudaEventDestroy(local_ev_[s]); local_ev_[s] = nullptr; }
        if (h2d_ev_[s])   { cudaEventDestroy(h2d_ev_[s]);   h2d_ev_[s] = nullptr; }
    }
    remote_init_ = false;
    for (VideoDecorder& d : vd) {
        for (uint8_t* p : d.r_y)  if (p) cudaFree(p);
        for (uint8_t* p : d.r_uv) if (p) cudaFree(p);
        for (uint8_t* p : d.h_y)  if (p) cudaFreeHost(p);
        for (uint8_t* p : d.h_uv) if (p) cudaFreeHost(p);
        d.r_y.clear(); d.r_uv.clear(); d.h_y.clear(); d.h_uv.clear();
    }
    // デコーダ（codec_ctx）はそれぞれ参照を持っているので、ここで外しても先に壊れない
    for (auto& [g, ctx] : remote_ctx_) if (ctx) av_buffer_unref(&ctx);
    remote_ctx_.clear();
}


// ==============================================================
// ★ リングのコマを壊さない受信
//
//   avcodec_receive_frame() は公式に
//     "the function will always call av_frame_unref(frame)
//      before doing anything else"
//   と定められている。EAGAIN / EOF で返るときも渡した AVFrame の
//   中身は先に捨てられるので、hw_frames[ringNo] を直接渡すと
//   投機的な receive が失敗しただけで取得済みのコマが消える。
//   必ずスクラッチへ受け、成功したときだけリングへ移すこと。
// ==============================================================
bool nvgpudecode::receive_into_slot(int i, int64_t* out_pts)
{
    if (i < 0 || i >= (int)vd.size())      return false;
    AVFrame* tmp = vd[i].tmp_frame;
    if (!tmp)                              return false;

    if (avcodec_receive_frame(vd[i].codec_ctx, tmp) != 0)
        return false;                       // ring は触っていない

    AVFrame* dst = vd[i].hw_frames[ringNo];
    av_frame_unref(dst);                    // move_ref は dst が空であることを要求する
    av_frame_move_ref(dst, tmp);

    if (out_pts) *out_pts = dst->best_effort_timestamp;
    return true;
}


// ==============================================================
// ★ 保留パケットの破棄
//   flush / シーク / 終了時に呼ぶ。残したまま flush すると、
//   シーク前のパケットをシーク後のデコーダへ入れてしまう。
// ==============================================================
void nvgpudecode::clear_pending()
{
    // ★ シーク/flush をまたいで見送り回数を持ち越さない
    mismatch_skips_ = 0;

    for (auto& q : pending_) {
        while (!q.empty()) {
            AVPacket* p = q.front();
            q.pop_front();
            av_packet_free(&p);
        }
    }
}


// ==============================================================
// ★ ファイルからパケットを 1 本読んで pending_ へ積む
//
//   ★★ ここでデコーダへは入れない。★★
//     デコーダに溜めると *_cuvid の出力サーフェス（自前プール、
//     既定 25 枚）を食い潰してデコードが止まる。
//     圧縮パケットをアプリ側で持つぶんには桁違いに軽い。
//
//   音声パケットはその場で処理して積まない。
//   戻り値 false = EOF
// ==============================================================
bool nvgpudecode::read_into_pending()
{
    if (pending_.size() != vd.size()) pending_.resize(vd.size());

    int ret = av_read_frame(fmt_ctx, packet);
    if (ret < 0) {
        av_packet_unref(packet);
        return false;                       // EOF
    }

    if (packet->stream_index == audio_stream_index) {
        get_decode_audio();
        av_packet_unref(packet);
        return true;
    }

    for (int i = 0; i < (int)vd.size(); i++) {
        if (packet->stream_index != vd[i].stream_index) continue;

        if ((int)pending_[i].size() >= kMaxPending) {
            // ★ ここまで溜まるのは揃えられていない証拠。放っておくと
            //   メモリが伸び続けるので、一番古いものから落とす。
            //   落とした区間は、そのタイルだけ次のキーフレームまで崩れる。
            warn_once_in_a_while(
                QString("[nvgpudecode] tile%1 の保留パケットが上限 %2 に達しました")
                    .arg(i).arg(kMaxPending));
            AVPacket* old = pending_[i].front();
            pending_[i].pop_front();
            av_packet_free(&old);
        }

        AVPacket* clone = av_packet_clone(packet);
        if (clone) pending_[i].push_back(clone);
        break;
    }

    av_packet_unref(packet);
    return true;
}


// ==============================================================
// ★ pending_[i] から 1 本取り出してデコーダへ入れ、出た絵を受け取る
//
//   ★ 送ったら必ず受け取ること（入出力 1:1）。
//     受け取らずに溜めると cuvid の出力プールが枯渇する。
//   ★ EAGAIN でパケットを捨てないこと。捨てるとそのタイルだけ
//     フレームが欠けて、以降ずっとタイル間がずれる。
//     （これが元々のズレの原因だった）
//
//   戻り値 true = このストリームからフレームが 1 枚出た
// ==============================================================
bool nvgpudecode::pump_stream(int i, std::vector<bool>& got,
                              std::vector<int64_t>& pts, int& got_count)
{
    if (i < 0 || i >= (int)pending_.size() || pending_[i].empty()) return false;

    AVPacket* p = pending_[i].front();

    QElapsedTimer tm;   // 性能 HUD の内訳（prof_）
    tm.start();
    int r = avcodec_send_packet(vd[i].codec_ctx, p);
    prof_.send[tileGpu(i)] += tm.nsecsElapsed() / 1e6;

    if (r == AVERROR(EAGAIN)) {
        // 出力キューが満杯。1 枚受け取って空けてから送り直す。
        // ★ ここで pending_ から取り除かないこと（＝捨てない）
        tm.restart();
        const bool ok = receive_into_slot(i, &pts[i]);
        prof_.recv[tileGpu(i)] += tm.nsecsElapsed() / 1e6;
        if (ok) {
            if (!got[i]) { got[i] = true; got_count++; }
            return true;
        }
        return false;
    }

    // 送れたので保留から外す
    pending_[i].pop_front();
    av_packet_free(&p);

    if (r < 0) return false;

    tm.restart();
    const bool ok = receive_into_slot(i, &pts[i]);
    prof_.recv[tileGpu(i)] += tm.nsecsElapsed() / 1e6;
    if (ok) {
        if (!got[i]) { got[i] = true; got_count++; }
        return true;
    }
    return false;                           // まだ出力遅延の途中
}


// ==============================================================
// ★ 警告の連打を防ぐ
//   同じ原因で毎フレーム出すと、肝心のログが流れて読めなくなる。
// ==============================================================
// ★ 呼び出し側は必ず warn_due() で先に判定すること。
//   文字列は呼ばれた時点で組み立てられるので、ここで間引いても
//   QString::arg() の確保は毎フレーム走ってしまう。
//   揃わないフレームが続くときは、まさにその確保が重なる。
bool nvgpudecode::warn_due()
{
    const int d = Frame.FrameNo - last_warn_frame_;
    return (d <= -60 || d >= 60);
}

void nvgpudecode::warn_once_in_a_while(const QString& msg)
{
    if (!warn_due()) return;
    last_warn_frame_ = Frame.FrameNo;
    qWarning().noquote() << msg;
}

// ★ 新しい絵が作れないときは前の Frame を送り直す。
//   in-flight を数えるのを忘れないこと。emit ごとに GUI から
//   receve_decode_flag が返って減算されるので、ここで足さないと
//   カウンタが下振れし、リング追い越しの検出が効かなくなる。
void nvgpudecode::resend_last_frame()
{
    if (Frame.d_y[0] == nullptr) return;
    ++inflight_frames;
    emit send_decode_image(Frame, true, video_reverse_flag);
}

// ==============================================================
// ★ タイル間の pts を揃える（max 基準・順再生のまま）
//
//   一番進んでいるタイルを基準に、「遅れているタイルだけ」を前へ進める。
//   シークはしない。
//
//   進め方は 2 段階:
//     ① デコーダ内部に溜まっている出力をまず引き出す
//        （出力遅延の差ぶんは、これだけで追いつく）
//     ② それでも足りなければ、そのストリームの pending_ から
//        パケットを入れて進める
//
//   ★ 他のタイルには触らない。全員へ一斉に送ると全員が進むので
//     差が縮まらない。読んだパケットは pending_ に積むだけで、
//     デコーダへは入れない（プールを食わない）。
//
//   ★ 揃っている間のコストは int64 の比較 N 回だけで、そのまま return する。
// ==============================================================
bool nvgpudecode::align_tiles_pts(std::vector<int64_t>& pts)
{
    const int N = (int)vd.size();
    if (N <= 1) return true;
    if ((int)pending_.size() != N) pending_.resize(N);

    // 許容誤差は半フレーム
    const int64_t tol = (VideoInfo.pts_per_frame > 1.0)
                            ? (int64_t)(VideoInfo.pts_per_frame / 2.0)
                            : 0;

    std::vector<bool> dummy_got(N, true);   // ここでは got を進めない
    int dummy_count = N;                    // pump_stream の参照引数を埋めるだけ

    for (int round = 0; round < kAlignMaxRounds; round++) {

        // ---- 基準 = 一番進んでいるタイル（max）----
        int64_t target = AV_NOPTS_VALUE;
        for (int i = 0; i < N; i++) {
            if (pts[i] == AV_NOPTS_VALUE) continue;
            if (target == AV_NOPTS_VALUE || pts[i] > target) target = pts[i];
        }
        if (target == AV_NOPTS_VALUE) return false;   // 1 枚も取れていない

        bool aligned   = true;
        bool advanced  = false;
        bool need_read = false;

        for (int i = 0; i < N; i++) {
            if (pts[i] != AV_NOPTS_VALUE && pts[i] + tol >= target) continue;

            // ① デコーダ内部の出力遅延ぶんを引き出す
            //   ★ 投機的に進めるので receive_into_slot を使うこと。
            //     直接 receive すると失敗時に取得済みのコマが消える。
            while (pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) {
                if (!receive_into_slot(i, &pts[i])) break;
                advanced = true;
            }

            // ② 足りなければ、このストリームにだけパケットを入れて進める
            int spin = 0;
            while ((pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) &&
                   !pending_[i].empty() && spin < kLoopGuard) {
                if (pump_stream(i, dummy_got, pts, dummy_count)) advanced = true;
                spin++;
            }

            if (pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) {
                aligned = false;
                if (pending_[i].empty()) need_read = true;
            }
        }

        if (aligned) return true;

        // 追いつかせる材料が無いなら読み足す（pending_ に積むだけ）
        if (need_read) {
            if (!read_into_pending()) return false;   // EOF
            continue;
        }

        if (!advanced) return false;        // これ以上進めない
    }

    return false;
}


// ==============================================================
// ★ タイル間 pts 不一致の検出
//
//   align_tiles_pts() で揃えているので通常は出ない。
//   出た場合は「同じ表示フレームに別の絵のタイルが混ざっている」ので、
//   絵が割れる。再発を取り逃がさないための保険。
// ==============================================================
bool nvgpudecode::check_tile_pts_mismatch()
{
    const int N = (int)vd.size();
    if (N <= 1) return true;

    const int64_t tol = (VideoInfo.pts_per_frame > 1.0)
                            ? (int64_t)(VideoInfo.pts_per_frame / 2.0)
                            : 0;

    const int64_t base = vd[0].hw_frames[ringNo]->best_effort_timestamp;

    for (int i = 1; i < N; i++) {
        const int64_t p = vd[i].hw_frames[ringNo]->best_effort_timestamp;
        const int64_t d = (p > base) ? (p - base) : (base - p);
        if (d > tol) {
            mismatch_total_++;
            // ★ warn_once_in_a_while は間引きが強すぎてこの検出を隠すので使わない
            if (mismatch_total_ <= 10 || (mismatch_total_ % 100) == 0) {
                qWarning().noquote()
                    << QString("[nvgpudecode] TILE PTS MISMATCH tile%1 pts=%2 vs tile0=%3 "
                               "(%4 frames) FrameNo=%5 累計=%6")
                           .arg(i).arg(p).arg(base)
                           .arg((double)(p - base) / VideoInfo.pts_per_frame, 0, 'f', 1)
                           .arg(Frame.FrameNo).arg(mismatch_total_);
            }
            return false;
        }
    }
    return true;
}


//映像シングルストリーム
void nvgpudecode::get_singledecode_image() {
    //----------- シーク処理 -----------
    if (slider_No != Frame.FrameNo || video_reverse_flag) {
        // ----------- 逆再生、通常シーク処理 -----------
        if (video_reverse_flag) {
            slider_No--;
            if (Frame.FrameNo-1 < VideoInfo.start_range_framesNo){
                high_res_seek_frame(Frame.FrameNo-1,false);
                return;
            }
        }

        avcodec_flush_buffers(vd[0].codec_ctx);
        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);

        av_seek_frame(fmt_ctx, vd[0].stream_index,
                      frame_to_pts(slider_No),
                      AVSEEK_FLAG_BACKWARD);

        Frame.FrameNo = slider_No;
        seek_flag = true;
    }else if(Frame.FrameNo+1 > VideoInfo.end_range_framesNo){
        // ----------- 範囲外シーク処理 -----------
        //終端、始端に戻す
        if(!video_reverse_flag){
            high_res_seek_frame(Frame.FrameNo+1,false);
            return;
        }
    }

    // ----------- パケット読み込みループ -----------
    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);

        // ---------- EOF ----------
        if (ret < 0) {

            avcodec_send_packet(vd[0].codec_ctx, nullptr);

            if (receive_into_slot(0)) {
                av_packet_unref(packet);
                break;
            }

            // 終端→先頭に戻ってループ
            if (Frame.FrameNo < VideoInfo.end_range_framesNo - 1) {
                avcodec_flush_buffers(vd[0].codec_ctx);
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                av_seek_frame(fmt_ctx, vd[0].stream_index,
                              frame_to_pts(Frame.FrameNo + 1),
                              AVSEEK_FLAG_ANY);
            } else {
                emit decode_end();
                high_res_seek_frame(VideoInfo.start_range_framesNo,false);
                // ★ high_res_seek_frame の中で既に1フレーム emit しているので、
                //   ここでループを抜けて二重 emit を避ける
                av_packet_unref(packet);
                return;
            }

            av_packet_unref(packet);
            continue;
        }

        // ----------- AUDIO PACKET -----------
        if (packet->stream_index == audio_stream_index) {
            get_decode_audio();
            av_packet_unref(packet);
            continue;
        }

        // ----------- VIDEO PACKET -----------
        if (packet->stream_index == vd[0].stream_index) {
            // ★ EAGAIN でパケットを捨てない（マルチと同じ理由）。
            //   単一ストリームではタイルズレにはならないが、
            //   そのぶんフレームが欠けてカクつきの原因になる。
            int r = avcodec_send_packet(vd[0].codec_ctx, packet);
            if (r == AVERROR(EAGAIN)) {
                if (receive_into_slot(0)) {
                    av_packet_unref(packet);
                    break;
                }
                r = avcodec_send_packet(vd[0].codec_ctx, packet);   // ★ 再送
            }
            if (r == 0) {
                if (receive_into_slot(0)) {
                    av_packet_unref(packet);
                    break;
                }
            }
        }
        av_packet_unref(packet);
    }

    publish_decode_frame();
    av_packet_unref(packet);
}

//映像マルチストリーム
void nvgpudecode::get_multidecode_image() {
    QElapsedTimer total;   // 性能 HUD の内訳（全体。残りを「その他」にする）
    total.start();
    prof_.clear();
    const int N = (int)vd.size();
    if ((int)pending_.size() != N) pending_.resize(N);

    std::vector<bool>    got_frame(N, false);
    std::vector<int64_t> pts(N, AV_NOPTS_VALUE);   // ★ 各タイルが掴んだ絵の pts
    int got_count = 0;

    //----------- シーク処理 -----------
    if (slider_No != Frame.FrameNo || video_reverse_flag) {
        // ----------- 逆再生、通常シーク処理 -----------
        if (video_reverse_flag) {
            slider_No--;
            if (Frame.FrameNo-1 < VideoInfo.start_range_framesNo){
                high_res_seek_frame(Frame.FrameNo-1,false);
                return;
            }
        }

        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);
        for (auto& dec : vd) {
            avcodec_flush_buffers(dec.codec_ctx);
        }
        clear_pending();          // ★ flush したら保留パケットも捨てる

        av_seek_frame(fmt_ctx, vd[0].stream_index,
                      frame_to_pts(slider_No),
                      AVSEEK_FLAG_BACKWARD);

        Frame.FrameNo = slider_No;
        seek_flag = true;
    }else if(Frame.FrameNo+1 > VideoInfo.end_range_framesNo){
        // ----------- 範囲外シーク処理 -----------
        //終端、始端に戻す
        if(!video_reverse_flag){
            high_res_seek_frame(Frame.FrameNo+1,false);
            return;
        }
    }

    // -------------------------------------------------------------
    // ★ 各ストリームから 1 枚ずつ取り出す。
    //
    //   パケットは pending_ に積んでから、まだフレームを取れていない
    //   ストリームにだけ投入する。全員へ一斉に送らないのが要点で、
    //   これによって「遅れているタイルだけを進める」ことができる。
    //   投入したら必ず受け取る（cuvid のプールを食わないため）。
    // -------------------------------------------------------------
    bool eof = false;
    // ★ 空回りガード。pump_stream が EAGAIN で送れず受け取れもしない、
    //   あるいは映像パケットがいつまでも来ない場合に無限ループしないよう、
    //   読み足し回数で上限を切る。
    int read_guard = 0;
    while (got_count < N && read_guard < kLoopGuard) {

        // ① 保留分から、まだ取れていないストリームを進める
        for (int i = 0; i < N; i++) {
            int spin = 0;
            while (!got_frame[i] && !pending_[i].empty() && spin < kLoopGuard) {
                pump_stream(i, got_frame, pts, got_count);
                spin++;
            }
        }
        if (got_count >= N) break;

        // ② 足りなければ読み足す（pending_ に積むだけ）
        QElapsedTimer tr;
        tr.start();
        const bool more = read_into_pending();
        prof_.read += tr.nsecsElapsed() / 1e6;
        if (!more) { eof = true; break; }
        read_guard++;
    }

    if (read_guard >= kLoopGuard) {
        warn_once_in_a_while(
            QString("[nvgpudecode] フレームを揃えられずに打ち切りました FrameNo=%1 (取得 %2/%3)")
                .arg(Frame.FrameNo).arg(got_count).arg(N));
    }

    // ----------- EOF 処理 -----------
    if (eof) {
        for (int i = 0; i < N; i++) {
            // ★ 取得済みのタイルには触らない（receive が消してしまう）
            if (got_frame[i]) continue;
            avcodec_send_packet(vd[i].codec_ctx, nullptr);
            if (receive_into_slot(i, &pts[i])) {
                got_frame[i] = true; got_count++;
            }
        }

        if (Frame.FrameNo >= VideoInfo.end_range_framesNo) {
            emit decode_end();
            high_res_seek_frame(VideoInfo.start_range_framesNo,false);
            // ★ high_res_seek_frame の中で既に1フレーム emit しているので抜ける
            return;
        }
    }

    // -------------------------------------------------------------
    // ★ タイル間の pts を揃える（順再生のまま。シークはしない）
    //
    //   揃わなかった場合もここでは何もしない。前フレームを保持したまま
    //   次のラウンドへ進めば、遅れているタイルだけが前進して自然に揃う。
    //   publish 側が null を弾くので、割れた絵は表示されない。
    // -------------------------------------------------------------
    if (!align_tiles_pts(pts)) {
        warn_once_in_a_while(
            QString("[nvgpudecode] タイルの pts がまだ揃っていません FrameNo=%1")
                .arg(Frame.FrameNo));
    }

    publish_decode_frame(&got_frame);
    emit_decode_breakdown(total.nsecsElapsed() / 1e6);
}

// 性能 HUD: デコード（待ち）の内訳を送る（表示用 GPU を先頭に、GPU ごとの送る・受け取る）
void nvgpudecode::emit_decode_breakdown(double totalMs)
{
    QVariantList parts;
    double sum = prof_.read + prof_.xfer;
    auto add = [&](const QString& key, const QString& label, double ms) {
        parts << QVariantMap{ {"key", key}, {"label", label}, {"ms", ms} };
    };
    add("read", QStringLiteral("読み込み"), prof_.read);
    std::vector<int> gpus{ g_openglDeviceID };
    for (const auto& [g, ms] : prof_.recv) if (g != g_openglDeviceID) gpus.push_back(g);
    for (int g : gpus) {
        const double s = prof_.send.count(g) ? prof_.send[g] : 0.0;
        const double r = prof_.recv.count(g) ? prof_.recv[g] : 0.0;
        if (!prof_.send.count(g) && !prof_.recv.count(g)) continue;
        add(QString("send%1").arg(g), QString("GPU%1 送る").arg(g), s);
        add(QString("recv%1").arg(g), QString("GPU%1 受取").arg(g), r);
        sum += s + r;
    }
    add("xfer",  QStringLiteral("転送の発行"), prof_.xfer);
    add("other", QStringLiteral("その他"), (std::max)(0.0, totalMs - sum));
    emit decode_breakdown(parts);
}

// ==============================================================
// ★ デコード結果の引き渡し（旧 CUDA_RGBA_to_merge）
//
//   ここではもう RGBA を作らない。
//   NVDEC が出力した NV12 タイルのポインタを VideoFrame に載せるだけ。
//   NV12→RGBA 結合は GUI スレッドが OpenGL テクスチャへ
//   surface object 経由で直接書き込む（GLWidget::uploadToGLTexture）。
//
//   ・★ cuvid は出力コピーを積むだけで同期せずに戻る。待ちは下で記録する
//     decode_ready イベントを使って GUI 側に取らせる。
//   ・タイルの寿命は vd[i].hw_frames のリングが一周するまで。
//     ringSize(=6) > GUI の先行許容(kMaxFramesInFlight=2) なので安全。
//     減らすときはこの関係を崩さないこと。
// ==============================================================
void nvgpudecode::publish_decode_frame(const std::vector<bool>* got){

    // -------------------------------------------------------------
    // ★ 1 タイルでもフレームを取れていなければ、新しい Frame を作らない。
    //
    //   hw_frames[ringNo] が埋まっていないタイルの data[0] は null で、
    //   そのまま GUI へ渡すと「入力データが null です」になる。
    //   前回の Frame を pause 扱いで再送し、ハンドシェイクだけ回す
    //   （emit しないと receve_decode_flag が来ず、再生が止まる）。
    // -------------------------------------------------------------
    if (got) {
        for (int i = 0; i < (int)vd.size() && i < (int)got->size(); i++) {
            if ((*got)[i]) continue;

            if (warn_due())
                warn_once_in_a_while(
                    QString("[nvgpudecode] tile%1 のフレームを取得できませんでした。"
                            "前フレームを再送します FrameNo=%2").arg(i).arg(Frame.FrameNo));

            resend_last_frame();
            return;
        }
    }

    // ★ 念のためポインタを直接確認する（上の got と二重だが、
    //   got を渡さない経路からの呼び出しもあるため）
    for (int i = 0; i < (int)vd.size() && i < VideoFrame::MAX_TILES; i++) {
        if (vd[i].hw_frames[ringNo]->data[0] == nullptr) {
            if (warn_due())
                warn_once_in_a_while(
                    QString("[nvgpudecode] tile%1 の中身が空です。前フレームを再送します")
                        .arg(i));
            resend_last_frame();
            return;
        }
    }

    // ★ タイル間の pts がずれていたらこのコマは出さない。
    //   エンコード中は割れた絵が出力ファイルに焼き付いて直せないため。
    //   前フレームを再送してハンドシェイクだけ回し、次ラウンドで揃え直す。
    //   連続 kMaxMismatchSkips 回で諦めて通す（再生が止まらないように）。
    if (!check_tile_pts_mismatch()) {
        if (mismatch_skips_ < kMaxMismatchSkips) {
            mismatch_skips_++;
            resend_last_frame();
            return;
        }
        qWarning().noquote()
            << QString("[nvgpudecode] タイルが連続 %1 回揃わないので、"
                       "割れたまま出力します FrameNo=%2")
                   .arg(kMaxMismatchSkips).arg(Frame.FrameNo);
        mismatch_skips_ = 0;
    } else {
        mismatch_skips_ = 0;
    }

    Frame.src_mode   = SRC_NV12_TILE;
    // ★ d_y/d_uv は MAX_TILES 本しかないので、本数もそこで頭打ちにする。
    //   ポインタを詰めるループだけ clamp されていて、本数が抜けていた。
    Frame.tile_count = std::min((int)vd.size(), (int)VideoFrame::MAX_TILES);
    Frame.tile_w     = VideoInfo.width;
    Frame.tile_h     = VideoInfo.height;

    std::vector<int> remoteTiles;   // このコマで別の GPU から送るタイル（transfer_remote）
    for (int i = 0; i < (int)vd.size() && i < VideoFrame::MAX_TILES; i++) {
        AVFrame* f = vd[i].hw_frames[ringNo];
        if (vd[i].gpu >= 0 && vd[i].gpu != g_openglDeviceID) {
            // 別の GPU でデコードしたタイル: 表示用 GPU のリングへ送ってから、そちらを渡す
            if (!alloc_remote_tile(i, f)) {
                resend_last_frame();
                return;
            }
            Frame.d_y[i]      = vd[i].r_y[ringNo];
            Frame.y_pitch[i]  = vd[i].r_ypitch;
            Frame.d_uv[i]     = vd[i].r_uv[ringNo];
            Frame.uv_pitch[i] = vd[i].r_uvpitch;
            remoteTiles.push_back(i);
            continue;
        }
        Frame.d_y[i]      = f->data[0];
        Frame.y_pitch[i]  = (size_t)f->linesize[0];
        Frame.d_uv[i]     = f->data[1];
        Frame.uv_pitch[i] = (size_t)f->linesize[1];
    }

    // フレーム番号と時刻（基底で共通化。0 除算と NOPTS もそこでふさいでいる）
    advance_frame_no(vd[0].hw_frames[ringNo]->best_effort_timestamp);

    // ==============================================================
    // ★ NVDEC の出力コピー完了を記録する
    //
    //   cuviddec.c の cuvid_output_frame は cuMemcpy2DAsync を積むだけで
    //   cuStreamSynchronize を呼ばずに戻る。したがって Frame.d_y が指す
    //   NV12 面はこの時点でまだ転送中。GUI 側に GPU 側で待たせる。
    //   ★ dec_stream は NULL（既定ストリーム）でも正常。判定は
    //     dec_sync_by_event で行うこと。
    // ==============================================================
    if (!remoteTiles.empty()) {
        // 別の GPU のタイルがあるコマ: 転送して decode_ready を記録（transfer_remote）
        QElapsedTimer tx;   // 性能 HUD の内訳（転送の発行）
        tx.start();
        const bool sent = transfer_remote(remoteTiles);
        prof_.xfer += tx.nsecsElapsed() / 1e6;
        if (!sent) {
            resend_last_frame();
            return;
        }
    } else if (dec_sync_by_event && ringNo < (int)decode_ready_ev.size() && decode_ready_ev[ringNo]) {
        if (cudaEventRecord(decode_ready_ev[ringNo], dec_stream) == cudaSuccess) {
            Frame.decode_ready = decode_ready_ev[ringNo];
        } else {
            cudaGetLastError();
            cudaDeviceSynchronize();
            Frame.decode_ready = nullptr;
        }
    } else {
        // フォールバック。初期化時に警告済み（同じコンテキストでなければ
        // これでは待てない）。
        cudaDeviceSynchronize();
        Frame.decode_ready = nullptr;
    }

    //リングを回す（次のデコードは別の面へ）
    ringNo++;
    if(ringNo>=ringSize) ringNo = 0;

    // ★ 追い越し検出：GUIが消費する前にリングを一周していないか
    {
        const int n = ++inflight_frames;
        if (n >= ringSize) {
            qWarning() << "[nvgpudecode] RING OVERRUN inflight =" << n
                       << "/" << ringSize << " FrameNo =" << Frame.FrameNo;
        }
    }

    emit send_decode_image(Frame,false,video_reverse_flag);
}

//高精度シーク
void nvgpudecode::high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)
{
    //0より低い、最大フレーム数より多い数値が来た場合は修正
    if(targetFrameNo<VideoInfo.start_range_framesNo){
        if(video_play_flag){
            targetFrameNo = VideoInfo.end_range_framesNo;
        }else{
            //停止時は始端固定 3、10、30秒送りの場合で範囲外の場合は始端に待機させる
            targetFrameNo = VideoInfo.start_range_framesNo;
        }
    }else if(targetFrameNo>VideoInfo.end_range_framesNo){
        if(video_play_flag){
            targetFrameNo = VideoInfo.start_range_framesNo;
        }else{
            //停止時は終端固定 3、10、30秒送りの場合で範囲外の場合は終端に待機させる
            targetFrameNo = VideoInfo.end_range_framesNo;
        }
    }else if(Frame.FrameNo < 0){
        //0以下の処理
        targetFrameNo = VideoInfo.end_range_framesNo;
    }
    // 重い処理が始まったらUIを止める
    if(heavy_UI_flag){
        //UIはfalse
        emit heavy_process_signal(false);
    }

    if(vd.size()==0){
        return;
    }else if(vd.size()==1){
        high_res_seek_frame_single(targetFrameNo);
    }else{
        high_res_seek_frame_multi(targetFrameNo);
    }

    // 重い処理が終わったらUIを戻す
    if(heavy_UI_flag){
        emit heavy_process_signal(true);
    }
}

//高精度シークシングルストリーム
void nvgpudecode::high_res_seek_frame_single(int targetFrameNo){
    // ----------- シーク処理 -----------
    avcodec_flush_buffers(vd[0].codec_ctx);
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    av_seek_frame(fmt_ctx, vd[0].stream_index,
                  frame_to_pts(targetFrameNo),
                  AVSEEK_FLAG_BACKWARD);

    // ----------- パケット読み込みループ -----------
    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }
    // ★ 終端に当たってデコーダも空、という状態を何回見たか。
    //   目標が実在しない位置だと、読み直し -> 終端 を永久に繰り返す。
    int eof_rounds = 0;

    while(!thread_stop_flag){
        // ★ 内側にも停止要求を見せる。外側だけだと、1 回のスキャンが
        //   長いファイルでクローズ時に待たされる。
        while (!thread_stop_flag) {
            int ret = av_read_frame(fmt_ctx, packet);

            // ---------- EOF ----------
            if (ret < 0) {
                avcodec_send_packet(vd[0].codec_ctx, nullptr);
                if (receive_into_slot(0)) {
                    av_packet_unref(packet);
                    break;
                }

                // ★ デコーダも空。これ以上このファイルから先へは進めない。
                //   1 回だけ読み直しを許し、それでも届かなければ諦める。
                if (++eof_rounds >= 2) {
                    av_packet_unref(packet);
                    publish_seek_fallback(targetFrameNo);
                    return;
                }

                // 終端→先頭に戻ってループ
                uint64_t seek_frame{};
                if (Frame.FrameNo < VideoInfo.end_range_framesNo - 1) {
                    seek_frame = Frame.FrameNo + 1;
                }

                avcodec_flush_buffers(vd[0].codec_ctx);
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                av_seek_frame(fmt_ctx, vd[0].stream_index,
                              frame_to_pts(seek_frame),
                              AVSEEK_FLAG_ANY);

                av_packet_unref(packet);
                continue;
            }

            // ----------- VIDEO PACKET -----------
            if (packet->stream_index == vd[0].stream_index) {
                // ★ EAGAIN でパケットを捨てない
                int r = avcodec_send_packet(vd[0].codec_ctx, packet);
                if (r == AVERROR(EAGAIN)) {
                    if (receive_into_slot(0)) {
                        av_packet_unref(packet);
                        break;
                    }
                    r = avcodec_send_packet(vd[0].codec_ctx, packet);   // ★ 再送
                }
                if (r == 0) {
                    if (receive_into_slot(0)) {
                        av_packet_unref(packet);
                        break;
                    }
                }
            }
            // ★ 映像以外（音声など）のパケットもここで解放する。
            //   シークは最後のキーフレームから目標まで読み続けるので、
            //   取りこぼすとその区間ぶん丸ごと漏れる。
            av_packet_unref(packet);
        }

        //1 つ手前のコマを覚えてから、今のコマの番号を出す
        back1FrameNo = FrameNo;
        FrameNo = pts_to_frame(vd[0].hw_frames[ringNo]->best_effort_timestamp);

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo <= FrameNo){
            publish_decode_frame();
            break;
        }
        av_packet_unref(packet);
    }
}

//高精度シークマルチストリーム
void nvgpudecode::high_res_seek_frame_multi(int targetFrameNo){
    const int N = (int)vd.size();
    if ((int)pending_.size() != N) pending_.resize(N);

    // ----------- シーク処理 -----------
    // ビデオ/オーディオの両方 flush
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    for (auto& dec : vd) {
        avcodec_flush_buffers(dec.codec_ctx);
    }
    clear_pending();              // ★ flush したら保留パケットも捨てる

    av_seek_frame(fmt_ctx, vd[0].stream_index,
                  frame_to_pts(targetFrameNo),
                  AVSEEK_FLAG_BACKWARD);

    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }

    while(!thread_stop_flag){
        std::vector<bool>    got_frame(N, false);
        std::vector<int64_t> pts(N, AV_NOPTS_VALUE);
        int got_count = 0;

        bool eof = false;
        int  read_guard = 0;
        while (got_count < N && read_guard < kLoopGuard) {
            for (int i = 0; i < N; i++) {
                int spin = 0;
                while (!got_frame[i] && !pending_[i].empty() && spin < kLoopGuard) {
                    pump_stream(i, got_frame, pts, got_count);
                    spin++;
                }
            }
            if (got_count >= N) break;
            if (!read_into_pending()) { eof = true; break; }
            read_guard++;
        }

        // -------------------------------------------------------------
        // ★ 終端では各デコーダが出力遅延ぶんのコマを抱えている
        //
        //   ファイルを読み切っても、*_cuvid の中には最後の数コマが残って
        //   いる（実測 3 コマ）。1 枚だけ受け取って終わると、そのぶん手前で
        //   止まる（実測: target=433 に対して全タイル 430 で停止）。
        //   ここで各タイルを「目標に届く」か「空になる」まで引き切る。
        //
        //   ★ 外側のループで少しずつ引き出す形にしてはいけない。
        //     リングの段は publish_decode_frame() でしか進まないので、
        //     全タイルが揃わないラウンドが 1 度でも挟まると、先に空へ
        //     なったタイルだけ古い絵が残って pts が食い違う。結果、
        //     1 ラウンドで止めるより手前へ戻る（実測: 2355 -> 2350）。
        //     引き切るのはこの 1 箇所で完結させること。
        //
        //   ★ 取得済みのタイルも対象にしてよい。receive_into_slot() は
        //     失敗時にリングを触らないので、空振りしても壊れない。
        // -------------------------------------------------------------
        //   ★ ここで一度に数コマ進むため、1 コマ戻しの行き先を
        //     「前ラウンドの番号」で済ませられない。タイル 0 が実際に
        //     通過したコマを控えておく（実測: 433 まで引き切ったのに
        //     戻し先が 429 のまま残っていた）。
        bool drain_tracked_back = false;
        if (eof) {
            // ★ 番号の算出は pts_to_frame() に通す。全タイルが同じ
            //   pts_per_frame を通るので、タイル間の関係は崩れない。
            const double ppf = VideoInfo.pts_per_frame;
            for (int i = 0; i < N; i++) {
                avcodec_send_packet(vd[i].codec_ctx, nullptr);
                for (;;) {
                    // 目標に届いているならそれ以上進めない（行き過ぎ防止）
                    if (got_frame[i] && ppf > 0.0 && pts[i] != AV_NOPTS_VALUE &&
                        pts_to_frame(pts[i]) >= targetFrameNo) break;

                    const int64_t before = pts[i];              // 上書き前の位置
                    if (!receive_into_slot(i, &pts[i])) break;   // デコーダが空
                    if (!got_frame[i]) { got_frame[i] = true; got_count++; }

                    if (i == 0 && before != AV_NOPTS_VALUE && ppf > 0.0) {
                        back1FrameNo = pts_to_frame(before);
                        drain_tracked_back = true;
                    }
                }
            }
        }

        // ★ シーク直後はタイル間のずれが一番出やすい。ここで必ず揃える。
        //   flush 直後なので各デコーダは空から始まっており、通常は揃う。
        align_tiles_pts(pts);

        //フレーム番号記憶
        //   ★ ドレインで実際の 1 つ手前を掴めた場合はそちらを優先する。
        if (!drain_tracked_back) back1FrameNo = FrameNo;
        FrameNo = pts_to_frame(vd[0].hw_frames[ringNo]->best_effort_timestamp);

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo<=FrameNo){
            publish_decode_frame(&got_frame);
            break;
        }

        if (eof) {
            // ★ 目標へ届かないまま終端。黙って抜けると
            //   「末尾へシークしたのに画が変わらない」になるので、
            //   取れている最後のコマを出す。
            publish_seek_fallback(targetFrameNo);
            break;
        }
    }
}

// =============================================================================
// ★ 届かなかったシークの後始末
//
//   ここへ来るのは end_range_framesNo が実際の最終フレームより後ろを
//   指しているとき。取れている最後のコマを出せば、スライダーは
//   publish_decode_frame() の中で実在する位置へ吸い付く。
//
//   原因（fps / pts_per_frame の取り違え か、ファイル側の尺の不一致）を
//   切り分けられるよう、判断に要る数値を 1 行にまとめて出しておく。
// =============================================================================
void nvgpudecode::publish_seek_fallback(int targetFrameNo)
{
    const int64_t ts   = vd[0].hw_frames[ringNo]->best_effort_timestamp;
    const double  ppf  = VideoInfo.pts_per_frame;
    const int     from = (ppf > 0.0 && ts != AV_NOPTS_VALUE) ? pts_to_frame(ts) : -1;

    // タイル別の到達位置。ここがバラバラなら出力遅延の差、
    // 揃っていればファイル側にそのコマが無い（または届く前に止まっている）。
    QString tiles;
    for (int i = 0; i < (int)vd.size(); i++) {
        const int64_t t = vd[i].hw_frames[ringNo]->best_effort_timestamp;
        tiles += QString(" t%1=%2").arg(i)
                     .arg((ppf > 0.0 && t != AV_NOPTS_VALUE) ? pts_to_frame(t) : -1);
    }

    qWarning().noquote() << QString(
        "[nvgpudecode] 終端へ届きませんでした"
        " target=%1 reached=%2 last_pts=%3"
        " end_range=%4 max=%5 pts_per_frame=%6 fps=%7%8")
        .arg(targetFrameNo).arg(from).arg((qlonglong)ts)
        .arg(VideoInfo.end_range_framesNo).arg(VideoInfo.max_framesNo)
        .arg(ppf).arg(VideoInfo.fps).arg(tiles);

    publish_decode_frame();
}
