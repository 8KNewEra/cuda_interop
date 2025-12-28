#include "save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    frame_index = 0;
    int ret = 0;

    //CUDAデバイスコンテキスト
    QString gpuId = QString::number(g_cudaDeviceID);
    ret = av_hwdevice_ctx_create(&hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
    if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

    // ① FormatContext は1回だけ
    ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, encode_settings.Save_Path.c_str());
    if (ret < 0 || !fmt_ctx) throw std::runtime_error("Failed to allocate format context");

    // ② Encoder / Stream を複数作る
    int max_size = encode_settings.encode_tile;
    if(max_size==1){
        height_ = h;
        width_  = w;
    }else if(max_size==4){
        height_ = h*1/2;
        width_  = w*1/2;
    }

    for (int i = 0; i < max_size; i++) {
        ve.emplace_back();

        initialized_ffmpeg_hardware_context(i);
        initialized_ffmpeg_codec_context(i,max_size);

        // ★ stream 作成だけ
        ve[i].stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!ve[i].stream) throw std::runtime_error("Failed to create stream");

        ve[i].stream->time_base = ve[i].codec_ctx->time_base;
        ret = avcodec_parameters_from_context(ve[i].stream->codecpar, ve[i].codec_ctx);
        if (ret < 0) throw std::runtime_error("Failed to copy codec parameters");

        ve[i].stream->time_base = tb;
        ve[i].stream->avg_frame_rate = fr;

        this->ve[i].stream = ve[i].stream;
    }

    // ③ ファイルオープンは1回
    ret = avio_open(&fmt_ctx->pb, encode_settings.Save_Path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) throw std::runtime_error("Failed to open output file");

    // ④ ヘッダ書き込みは全 stream 作成後
    ret = avformat_write_header(fmt_ctx, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to write header");

    //Stream作成
    cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking
        );

    //event作成
    cudaEventCreateWithFlags(
        &event,
        cudaEventDisableTiming
        );
}

save_encode::~save_encode() {
    for(int i=0;i<ve.size();i++){
        //全フレーム送信後にフラッシュ
        avcodec_send_frame(ve[i].codec_ctx, nullptr);

        AVPacket *pkt = av_packet_alloc();
        while (true) {
            int ret = avcodec_receive_packet(ve[i].codec_ctx, pkt);
            if (ret == AVERROR(EAGAIN)) {
                continue;
            }
            if (ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                throw std::runtime_error("Error during flushing");
            }

            av_packet_rescale_ts(pkt, ve[i].codec_ctx->time_base, ve[i].stream->time_base);
            pkt->stream_index = ve[i].stream->index;
            av_interleaved_write_frame(fmt_ctx, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);

        //ハードウェアフレームを解放
        if (ve[i].hw_frame) {
            av_frame_free(&ve[i].hw_frame);
            ve[i].hw_frame = nullptr;
        }
        //ハードウェアフレームコンテキストを解放
        if (ve[i].hw_frames_ctx) {
            av_buffer_unref(&ve[i].hw_frames_ctx);
            ve[i].hw_frames_ctx = nullptr;
        }
        //コーデックコンテキストを解放
        if (ve[i].codec_ctx) {
            avcodec_free_context(&ve[i].codec_ctx);
            ve[i].codec_ctx = nullptr;
        }
    }

    //ハードウェアデバイスコンテキストを解放
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
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

    //Stream削除
    if(stream){
        cudaStreamDestroy(stream);
        stream=nullptr;
    }

    //event削除
    if (event) {
        cudaEventDestroy(event);
        event = nullptr;
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "save_encode: Destructor called";
}

void save_encode::initialized_ffmpeg_codec_context(int i,int max_split){
    //エンコーダの取得とコンテキスト作成
    const AVCodec* codec = avcodec_find_encoder_by_name(encode_settings.Codec.c_str());
    if (!codec) throw std::runtime_error("hevc_nvenc codec not found");

    ve[i].codec_ctx = avcodec_alloc_context3(codec);
    if (!ve[i].codec_ctx) throw std::runtime_error("Failed to allocate AVCodecContext");

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

    ve[i].codec_ctx->width = width_;
    ve[i].codec_ctx->height = height_;
    ve[i].codec_ctx->pix_fmt = hw_pix_fmt;

    fps = encode_settings.save_fps;
    pts_step = 15000 / 60;
    fr = { fps, 1 };
    tb = { 1, fps * pts_step };
    ve[i].codec_ctx->time_base = tb;
    ve[i].codec_ctx->framerate = fr;

    //初期化された hw_* を参照
    ve[i].codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    ve[i].codec_ctx->hw_frames_ctx = av_buffer_ref(ve[i].hw_frames_ctx);

    //ビットレート周りの設定
    AVDictionary* opts = nullptr;

    // cbr, vbr, cq 切り替え
    if (encode_settings.rc_mode == "cq") {
        // --- CQ モード ---
        av_dict_set_int(&opts, "cq", encode_settings.cq, 0);
    } else if (encode_settings.rc_mode == "vbr") {
        // --- VBR モード ---
        ve[i].codec_ctx->bit_rate = encode_settings.target_bit_rate;
        ve[i].codec_ctx->rc_max_rate = encode_settings.max_bit_rate;
        ve[i].codec_ctx->rc_buffer_size = encode_settings.max_bit_rate;
    } else if (encode_settings.rc_mode == "cbr") {
        // --- CBR モード ---
        ve[i].codec_ctx->bit_rate = encode_settings.target_bit_rate;
        ve[i].codec_ctx->rc_max_rate = encode_settings.max_bit_rate;
        ve[i].codec_ctx->rc_buffer_size = encode_settings.target_bit_rate;
    }

    // 共通オプション
    av_dict_set(&opts, "preset", encode_settings.preset.c_str(), 0);
    av_dict_set(&opts, "tune", encode_settings.tune.c_str(), 0);
    av_dict_set_int(&opts, "g", encode_settings.gop_size, 0);
    av_dict_set_int(&opts, "bf", encode_settings.b_frames, 0);

    //分割エンコード使用時
    if(max_split>1){
        av_dict_set(&opts, "rc-lookahead", "0", 0); // lookahead 無効
        av_dict_set(&opts, "delay", "0", 0);
        av_dict_set(&opts, "zerolatency", "1", 0);
        av_dict_set(&opts, "async_depth", "1", 0);
    }

    if(encode_settings.split_encode_mode=="0"){
        av_dict_set_int(&opts, "split_encode_mode", 0, 0);
    }else{
        av_dict_set_int(&opts, "split_encode_mode", 3, 0);
    }


    // 1pass / 2pass 切り替え
    qDebug()<<encode_settings.pass_mode;
    if (encode_settings.pass_mode == "2pass-quarter-res") {
        av_dict_set_int(&opts, "multipass", 1, 0);
    } else if (encode_settings.pass_mode == "2pass-full-res") {
        av_dict_set_int(&opts, "multipass", 2, 0);
        qDebug()<<"full";
    } else {
        av_dict_set_int(&opts, "multipass", 0, 0);
    }

    int ret = avcodec_open2(ve[i].codec_ctx, codec, &opts);
    if (ret < 0) throw std::runtime_error("Failed to open codec");
    av_dict_free(&opts);
}

void save_encode::initialized_ffmpeg_hardware_context(int i) {
    int ret = 0;

    ve[i].hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!ve[i].hw_frames_ctx){
        throw std::runtime_error("Failed to allocate hw_frames_ctx");
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(ve[i].hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width_;
    frames_ctx->height = height_;
    frames_ctx->initial_pool_size = 10;
    ret = av_hwframe_ctx_init(ve[i].hw_frames_ctx);
    if (ret < 0) throw std::runtime_error("Failed to init frames_ctx");

    ve[i].hw_frame = av_frame_alloc();
    ve[i].hw_frame->format = AV_PIX_FMT_CUDA;
    ve[i].hw_frame->width = width_;
    ve[i].hw_frame->height = height_;
    ret = av_hwframe_get_buffer(ve[i].hw_frames_ctx, ve[i].hw_frame, 0);
    if (ret < 0) throw std::runtime_error("Failed to alloc hw_frame");
}

void save_encode::encode(uint8_t* d_rgba, size_t pitch_rgba){
    if(encode_settings.encode_tile==1){
        normal_encode(d_rgba,pitch_rgba);
    }else if(encode_settings.encode_tile==4){
        encode_split_x4(d_rgba,pitch_rgba);
    }
}

bool save_encode::encode_split_x4(uint8_t* d_rgba, size_t pitch_rgba)
{
    //qDebug()<<No;
    No+=1;

    //RGBAをNV12に変換してffmpegへ転送
    CUDA_IMG_Proc->rgba_to_nv12x4_flip_split(
        d_rgba,pitch_rgba,
        ve[0].hw_frame->data[0],ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1],ve[0].hw_frame->linesize[1],
        ve[1].hw_frame->data[0],ve[1].hw_frame->linesize[0], ve[1].hw_frame->data[1],ve[1].hw_frame->linesize[1],
        ve[2].hw_frame->data[0],ve[2].hw_frame->linesize[0], ve[2].hw_frame->data[1],ve[2].hw_frame->linesize[1],
        ve[3].hw_frame->data[0],ve[3].hw_frame->linesize[0], ve[3].hw_frame->data[1],ve[3].hw_frame->linesize[1],
        width_*2,height_*2,width_,height_,stream);

    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    // 1. 全 encoder に frame を送る
    for (int i = 0; i < ve.size(); i++) {
        ve[i].hw_frame->pts = frame_index * pts_step;
        avcodec_send_frame(ve[i].codec_ctx, ve[i].hw_frame);
    }

    // 2. 全 encoder から packet を回収
    bool got;
    do {
        got = false;
        for (int i = 0; i < ve.size(); i++) {
            AVPacket* pkt = av_packet_alloc();
            int ret = avcodec_receive_packet(ve[i].codec_ctx, pkt);
            if (ret == 0) {
                av_packet_rescale_ts(pkt,
                                     ve[i].codec_ctx->time_base,
                                     ve[i].stream->time_base);
                pkt->stream_index = ve[i].stream->index;
                av_interleaved_write_frame(fmt_ctx, pkt);
                got = true;
            }
            av_packet_free(&pkt);
        }
    } while (got);

    frame_index+=1;
    //qDebug()<<frame_index;

    return true;
}

bool save_encode::normal_encode(uint8_t* d_rgba, size_t pitch_rgba)
{
    //qDebug()<<No;
    No+=1;

    //RGBAをNV12に変換してffmpegへ転送
    CUDA_IMG_Proc->Flip_RGBA_to_NV12(ve[0].hw_frame->data[0], ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1], ve[0].hw_frame->linesize[1],d_rgba, pitch_rgba,height_, width_,stream);

    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    // 1. 全 encoder に frame を送る
    ve[0].hw_frame->pts = frame_index * pts_step;
    avcodec_send_frame(ve[0].codec_ctx, ve[0].hw_frame);

    // 2. 全 encoder から packet を回収
    bool got;
    do {
        got = false;
        AVPacket* pkt = av_packet_alloc();
        int ret = avcodec_receive_packet(ve[0].codec_ctx, pkt);
        if (ret == 0) {
            av_packet_rescale_ts(pkt,
                                 ve[0].codec_ctx->time_base,
                                 ve[0].stream->time_base);
            pkt->stream_index = ve[0].stream->index;
            av_interleaved_write_frame(fmt_ctx, pkt);
            got = true;
            av_packet_free(&pkt);
        }
    } while (got);

    frame_index+=1;
    //qDebug()<<frame_index;

    return true;
}
