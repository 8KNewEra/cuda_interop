#include "save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    height_=h;
    width_=w;
    frame_index=0;

    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }

    // qDebug()<<encode_settings.Codec;
    // qDebug()<<encode_settings.gop_size;
    // qDebug()<<encode_settings.b_frames;
    // qDebug()<<encode_settings.split_encode_mode;
    // qDebug()<<encode_settings.pass_mode;
    // qDebug()<<encode_settings.rc_mode;
    // qDebug()<<encode_settings.preset;
    // qDebug()<<encode_settings.tune;
    // qDebug()<<encode_settings.save_fps;
    // qDebug()<<encode_settings.target_bit_rate;
    // qDebug()<<encode_settings.max_bit_rate;
    // qDebug()<<encode_settings.crf;

    //CUDA関連初期化（先に必要な ctx ができる）
    initialized_ffmpeg_hardware_context();
    initialized_ffmpeg_codec_context();
    initialized_ffmpeg_output(encode_settings.Save_Path);
}

save_encode::~save_encode() {
    //全フレーム送信後にフラッシュ
    avcodec_send_frame(codec_ctx, nullptr);

    AVPacket *pkt = av_packet_alloc();
    while (true) {
        int ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN)) {
            continue;
        }
        if (ret == AVERROR_EOF) {
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

    //ハードウェアフレームを解放
    if (hw_frame) {
        av_frame_free(&hw_frame);
        hw_frame = nullptr;
    }
    //ハードウェアフレームコンテキストを解放
    if (hw_frames_ctx) {
        av_buffer_unref(&hw_frames_ctx);
        hw_frames_ctx = nullptr;
    }
    //ハードウェアデバイスコンテキストを解放
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
    }

    //コーデックコンテキストを解放
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
        codec_ctx = nullptr;
    }

    //フォーマットコンテキストと出力ファイルを解放
    if (fmt_ctx) {
        av_write_trailer(fmt_ctx);
        if (fmt_ctx->pb) {
            avio_closep(&fmt_ctx->pb);
        }
        avformat_free_context(fmt_ctx);
        fmt_ctx = nullptr;
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "save_encode: Destructor called";
}

void save_encode::initialized_ffmpeg_codec_context(){
    //エンコーダの取得とコンテキスト作成
    const AVCodec* codec = avcodec_find_encoder_by_name(encode_settings.Codec.c_str());
    if (!codec) throw std::runtime_error("hevc_nvenc codec not found");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) throw std::runtime_error("Failed to allocate AVCodecContext");

    //これは codec_ctx->pix_fmt に設定するものです
    enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
    for (int i = 0; ; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
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

    codec_ctx->width = width_;
    codec_ctx->height = height_;
    codec_ctx->pix_fmt = hw_pix_fmt;

    fps = encode_settings.save_fps;
    pts_step = 15000 / 60;
    fr = { fps, 1 };
    tb = { 1, fps * pts_step };
    codec_ctx->time_base = tb;
    codec_ctx->framerate = fr;

    //初期化された hw_* を参照
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    //ビットレート周りの設定
    AVDictionary* opts = nullptr;

    // cbr, vbr, cq 切り替え
    if (encode_settings.rc_mode == "cq") {
        // --- CQ モード ---
        av_dict_set_int(&opts, "cq", encode_settings.cq, 0);
    } else if (encode_settings.rc_mode == "vbr") {
        // --- VBR モード ---
        codec_ctx->bit_rate = encode_settings.target_bit_rate;
        codec_ctx->rc_max_rate = encode_settings.max_bit_rate;
        codec_ctx->rc_buffer_size = encode_settings.max_bit_rate;
    } else if (encode_settings.rc_mode == "cbr") {
        // --- CBR モード ---
        codec_ctx->bit_rate = encode_settings.target_bit_rate;
        codec_ctx->rc_max_rate = encode_settings.max_bit_rate;
        codec_ctx->rc_buffer_size = encode_settings.target_bit_rate;
    }

    // 共通オプション
    av_dict_set(&opts, "preset", encode_settings.preset.c_str(), 0);
    av_dict_set(&opts, "tune", encode_settings.tune.c_str(), 0);
    av_dict_set_int(&opts, "g", encode_settings.gop_size, 0);
    av_dict_set_int(&opts, "bf", encode_settings.b_frames, 0);
    av_dict_set_int(&opts, "split_encode_mode", 3, 0);

    // 1pass / 2pass 切り替え
    if (encode_settings.pass_mode == "2pass-quarter") {
        av_dict_set(&opts, "multi_pass", "2pass-quarter-res", 0);
    } else if (encode_settings.pass_mode == "2pass-full") {
        av_dict_set(&opts, "multi_pass", "2pass-full-res", 0);
    } else {
        av_dict_set(&opts, "multi_pass", "1pass", 0);
    }

    int ret = avcodec_open2(codec_ctx, codec, &opts);
    if (ret < 0) throw std::runtime_error("Failed to open codec");
    av_dict_free(&opts);
}

void save_encode::initialized_ffmpeg_hardware_context() {
    int ret = 0;

    // CUDA デバイスコンテキスト作成
    QString gpuId = QString::number(g_cudaDeviceID);
    ret = av_hwdevice_ctx_create(&hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
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

void save_encode::initialized_ffmpeg_output(const std::string& path){
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

    this->stream = stream;
}

bool save_encode::encode(uint8_t* d_rgba, size_t pitch_rgba)
{
    //qDebug()<<No;
    No+=1;

    //RGBAをNV12に変換してffmpegへ転送
    CUDA_IMG_Proc->Flip_RGBA_to_NV12(hw_frame->data[0], hw_frame->linesize[0], hw_frame->data[1], hw_frame->linesize[1],d_rgba, pitch_rgba,height_, width_);

    //フレームのPTSをセット
    hw_frame->pts = frame_index*pts_step;
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
