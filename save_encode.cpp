#include "save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    height_=h;
    width_=w;
    frame_index=0;
    fps = 120;
    pts_step = 15000/60;

    // 1. codec_ctxをnullptrで初期化
    codec_ctx = nullptr;

    // 2. まず CUDA 関連初期化（先に必要な ctx ができる）
    initialized_ffmpeg();

    // 3. エンコーダの取得とコンテキスト作成
    const AVCodec* codec = avcodec_find_encoder_by_name("av1_nvenc");
    if (!codec) throw std::runtime_error("hevc_nvenc codec not found");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) throw std::runtime_error("Failed to allocate AVCodecContext");

    // (これは codec_ctx->pix_fmt に設定するものです)
    enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
    for (int i = 0; ; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
        if (!config) {
            fprintf(stderr, "Encoder %s does not support any hardware config.\n", codec->name);
            break; // エラー
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX ) {
            hw_pix_fmt = config->pix_fmt; // この config->pix_fmt が NVENC が実際に期待するフォーマットです
            break;
        }
    }
    if (hw_pix_fmt == AV_PIX_FMT_NONE) {
        fprintf(stderr, "No suitable hardware pixel format found for encoder %s.\n", codec->name);
        // エラーハンドリング
        // ここでエラーになる場合、NVENCがNV12を期待していない可能性も
        // 代わりに AV_PIX_FMT_NV12 を直接指定する必要があるか、
        // または別のピクセルフォーマットで試す必要があるかもしれません。
    }

    codec_ctx->width = width_;
    codec_ctx->height = height_;
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;

    fr = {fps, 1};
    tb = {1, fps * pts_step};
    codec_ctx->time_base = tb;
    codec_ctx->framerate = fr;
    codec_ctx->gop_size =30;
    codec_ctx->max_b_frames = 0;
    codec_ctx->bit_rate = 200 * 1000 * 1000;

    // 4. 正しく初期化された hw_* を参照
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    //Split Frame Encode
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "split_encode_mode", "1", 0);
    av_dict_set(&opts, "async_depth", "4", 0);

    int ret = avcodec_open2(codec_ctx, codec, &opts);
    if (ret < 0) throw std::runtime_error("Failed to open codec");
    av_dict_free(&opts);

    initialized_output("D:/test2.mp4");
}

save_encode::~save_encode() {
    // 全フレーム送信後にフラッシュ
    avcodec_send_frame(codec_ctx, nullptr); // NULL送信でフラッシュ開始

    AVPacket *pkt = av_packet_alloc();
    while (true) {
        int ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN)) {
            // 内部処理が残っている可能性 → 続行
            continue;
        }
        if (ret == AVERROR_EOF) {
            // すべてのパケットが出力済み
            break;
        }
        if (ret < 0) {
            throw std::runtime_error("Error during flushing");
        }

        av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;
        av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);



    // FFmpegリソースの解放
    // 重要なのは、割り当てられた順序と逆順に解放することと、
    // 親となるコンテキストを解放する前に子を解放することです。
    // 例: avformat_free_contextより前にavcodec_free_context。

    // 最初にハードウェアフレームを解放
    if (hw_frame) {
        av_frame_free(&hw_frame);
        hw_frame = nullptr;
    }
    // ハードウェアフレームコンテキストを解放
    if (hw_frames_ctx) {
        av_buffer_unref(&hw_frames_ctx);
        hw_frames_ctx = nullptr;
    }
    // ハードウェアデバイスコンテキストを解放
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
    }

    // コーデックコンテキストを解放
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
        codec_ctx = nullptr;
    }

    // フォーマットコンテキストと出力ファイルを解放
    if (fmt_ctx) {
        // ファイルフッターの書き込みは、すべてのパケットが書き込まれた後に行う
        // ただし、fmt_ctx->pb がオープンしている場合のみ
        av_write_trailer(fmt_ctx); // フッター書き込み
        if (fmt_ctx->pb) { // I/Oコンテキストがオープンしているか確認
            avio_closep(&fmt_ctx->pb); // 出力ファイルを閉じる
        }
        avformat_free_context(fmt_ctx); // フォーマットコンテキストを解放
        fmt_ctx = nullptr;
    }

    qDebug() << "save_encode: Destructor called";
}

void save_encode::initialized_ffmpeg() {
    int ret = 0;

    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

    hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ctx){
        throw std::runtime_error("Failed to allocate hw_frames_ctx");
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width_;
    frames_ctx->height = height_;
    frames_ctx->initial_pool_size = 10;
    ret = av_hwframe_ctx_init(hw_frames_ctx);
    if (ret < 0) throw std::runtime_error("Failed to init frames_ctx");

    hw_frame = av_frame_alloc();
    hw_frame->format = AV_PIX_FMT_CUDA;
    hw_frame->width = width_;
    hw_frame->height = height_;
    ret = av_hwframe_get_buffer(hw_frames_ctx, hw_frame, 0);
    if (ret < 0) throw std::runtime_error("Failed to alloc hw_frame");
}

void save_encode::initialized_output(const std::string& path){
    int ret = 0;

    ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, path.c_str());
    if (ret < 0 || !fmt_ctx) throw std::runtime_error("Failed to allocate format context");

    stream = avformat_new_stream(fmt_ctx, nullptr);
    if (!stream) throw std::runtime_error("Failed to create stream");

    stream->time_base = codec_ctx->time_base;
    ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
    if (ret < 0) throw std::runtime_error("Failed to copy codec parameters");

    ret = avio_open(&fmt_ctx->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) throw std::runtime_error("Failed to open output file");

    ret = avformat_write_header(fmt_ctx, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to write header");

    stream->time_base = tb;
    stream->avg_frame_rate = fr;

    this->stream = stream; // （必要であれば）
}

bool save_encode::encode(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv)
{
    //qDebug()<<No;
    No+=1;
    if(No==1){
        return false;
    }

    //ffmpegへ転送
    cudaMemcpy2D(
        hw_frame->data[0], hw_frame->linesize[0],
        d_y, pitch_y,
        width_ * sizeof(uint8_t), height_,
        cudaMemcpyDeviceToDevice
        );

    cudaMemcpy2D(
        hw_frame->data[1], hw_frame->linesize[1],
        d_uv, pitch_uv,
        width_ * sizeof(uint8_t), height_ / 2,
        cudaMemcpyDeviceToDevice
        );

    //フレームのPTS（表示時間）などをセット
    hw_frame->pts = frame_index*pts_step;  // 0,1,2,...
    frame_index+=1;

    //エンコーダにフレーム送信
    int ret = avcodec_send_frame(codec_ctx, hw_frame);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame to encoder");
    }

    AVPacket* pkt = av_packet_alloc();
    while (true) {
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            throw std::runtime_error("Error receiving packet from encoder");
        }

        av_packet_rescale_ts(pkt, codec_ctx->time_base, stream->time_base);
        pkt->stream_index = stream->index;

        ret = av_interleaved_write_frame(fmt_ctx, pkt);
        if (ret < 0) {
            throw std::runtime_error("Error writing packet to output");
        }

        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    //qDebug()<<frame_index;

    return true;
}
