#include "save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    width_=w;
    height_=h;
    frame_index = 0;
    int ret = 0;
    packet = av_packet_alloc();

    //CUDAデバイスコンテキスト
    QString gpuId = QString::number(g_cudaDeviceID);
    ret = av_hwdevice_ctx_create(&hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
    if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

    // ① FormatContext は1回だけ
    ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, encode_settings.Save_Path.c_str());
    if (ret < 0 || !fmt_ctx) throw std::runtime_error("Failed to allocate format context");

    // ② Encoder / Stream を複数作る
    for (int i = 0; i < encode_settings.encode_tile; i++) {
        ve.emplace_back();

        initialized_ffmpeg_hardware_context(i);
        initialized_ffmpeg_codec_context(i,encode_settings.encode_tile);

        // ★ stream 作成だけ
        ve[i].stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!ve[i].stream) throw std::runtime_error("Failed to create stream");

        ve[i].stream->time_base = ve[i].codec_ctx->time_base;
        ve[i].stream->avg_frame_rate = ve[i].codec_ctx->framerate;
        ret = avcodec_parameters_from_context(ve[i].stream->codecpar, ve[i].codec_ctx);
        if (ret < 0) throw std::runtime_error("Failed to copy codec parameters");

        this->ve[i].stream = ve[i].stream;
    }

    init_audio_encoder();

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
    // ==========================
    // Video flush（NVENC）
    // ==========================
    for (int i = 0; i < ve.size(); i++) {
        // flush
        avcodec_send_frame(ve[i].codec_ctx, nullptr);
        while (true) {
            int ret = avcodec_receive_packet(ve[i].codec_ctx, packet);

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(packet);
                break;
            }

            av_packet_rescale_ts(
                 packet,
                ve[i].codec_ctx->time_base,
                ve[i].stream->time_base
                );

            packet->stream_index = ve[i].stream->index;
            // ★ ソートしない ★
            av_interleaved_write_frame(fmt_ctx, packet);
            av_packet_unref(packet);
        }
    }

    // ==========================
    // Audio flush（AAC）
    // ==========================
    const int fs = audio_enc_ctx->frame_size;
    int remain = av_audio_fifo_size(audio_fifo);

    if (remain > 0) {
        AVFrame* f = av_frame_alloc();
        f->nb_samples = fs;
        f->format = audio_enc_ctx->sample_fmt;
        f->sample_rate = audio_enc_ctx->sample_rate;
        av_channel_layout_copy(&f->ch_layout, &audio_enc_ctx->ch_layout);

        av_frame_get_buffer(f, 0);

        av_audio_fifo_read(audio_fifo, (void**)f->data, remain);

        av_samples_set_silence(
            f->data,
            remain,
            fs - remain,
            audio_enc_ctx->ch_layout.nb_channels,
            audio_enc_ctx->sample_fmt
            );

        f->pts = audio_pts;
        audio_pts += fs;

        avcodec_send_frame(audio_enc_ctx, f);
        av_frame_free(&f);
    }

    avcodec_send_frame(audio_enc_ctx, nullptr);

    // ---- 各メモリ解放 ----
    for(int i=0;i<ve.size();i++){
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

    //パケット開放
    if(packet){
        av_packet_free(&packet);
        packet = nullptr;
    }

    if (audio_enc_ctx){
        avcodec_free_context(&audio_enc_ctx);
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

    //メタデータ設定
    ve[i].codec_ctx->width = width_/encode_settings.width_tile;
    ve[i].codec_ctx->height = height_/encode_settings.height_tile;
    ve[i].codec_ctx->pix_fmt = hw_pix_fmt;
    ve[i].codec_ctx->time_base = AVRational{1, encode_settings.save_fps};
    ve[i].codec_ctx->framerate = AVRational{encode_settings.save_fps, 1};

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
    frames_ctx->width = width_/encode_settings.width_tile;
    frames_ctx->height = height_/encode_settings.height_tile;
    frames_ctx->initial_pool_size = 10;
    ret = av_hwframe_ctx_init(ve[i].hw_frames_ctx);
    if (ret < 0) throw std::runtime_error("Failed to init frames_ctx");

    ve[i].hw_frame = av_frame_alloc();
    ve[i].hw_frame->format = AV_PIX_FMT_CUDA;
    ve[i].hw_frame->width = width_/encode_settings.width_tile;
    ve[i].hw_frame->height = height_/encode_settings.height_tile;
    ret = av_hwframe_get_buffer(ve[i].hw_frames_ctx, ve[i].hw_frame, 0);
    if (ret < 0) throw std::runtime_error("Failed to alloc hw_frame");
}

void save_encode::init_audio_encoder()
{
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec)
        throw std::runtime_error("AAC encoder not found");

    audio_enc_ctx = avcodec_alloc_context3(codec);

    // ---- encoder 設定 ----
    audio_enc_ctx->sample_rate = VideoInfo.in_sample_rate;              // ★ decode 側もこれに合わせる
    audio_enc_ctx->bit_rate    = 192000;
    audio_enc_ctx->time_base   = AVRational{1, audio_enc_ctx->sample_rate};

    av_channel_layout_default(&audio_enc_ctx->ch_layout, 2);
    audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP; // AAC native

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(audio_enc_ctx, codec, nullptr) < 0)
        throw std::runtime_error("avcodec_open2(audio) failed");

    if (audio_enc_ctx->frame_size != 1024)
        throw std::runtime_error("Unexpected AAC frame_size");

    // ---- stream ----
    audio_stream = avformat_new_stream(fmt_ctx, nullptr);
    if (!audio_stream)
        throw std::runtime_error("avformat_new_stream(audio) failed");

    audio_stream->time_base = audio_enc_ctx->time_base;

    if (avcodec_parameters_from_context(
            audio_stream->codecpar,
            audio_enc_ctx) < 0)
        throw std::runtime_error("copy audio params failed");

    // ---- FIFO ----
    audio_fifo = av_audio_fifo_alloc(
        audio_enc_ctx->sample_fmt,
        audio_enc_ctx->ch_layout.nb_channels,
        4096
        );
    if (!audio_fifo)
        throw std::runtime_error("av_audio_fifo_alloc failed");

    // ---- swr_enc (S16 → FLTP) ----
    int ret = swr_alloc_set_opts2(
        &swr_enc,
        &audio_enc_ctx->ch_layout,
        audio_enc_ctx->sample_fmt,   // FLTP
        audio_enc_ctx->sample_rate,
        &audio_enc_ctx->ch_layout,
        VideoInfo.out_format,            // decode PCM
        audio_enc_ctx->sample_rate,
        0,
        nullptr
        );

    if (ret < 0 || !swr_enc)
        throw std::runtime_error("swr_alloc_set_opts2 failed");

    if (swr_init(swr_enc) < 0)
        throw std::runtime_error("swr_init (encoder) failed");

    // ---- pts 初期化 ----
    audio_pts = 0;
}


void save_encode::encode(VideoFrame Frame){
    // ------------------------
    // CUDA NV12変換
    // ------------------------
    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    //本カーネル
    if(ve.size()==1){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->Flip_RGBA_to_NV12(ve[0].hw_frame->data[0], ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1], ve[0].hw_frame->linesize[1],Frame.d_encode_rgba, Frame.encode_pitch,width_,height_,stream);
    }else if(ve.size()==2){
        CUDA_IMG_Proc->rgba_to_nv12x2_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].hw_frame->data[0],ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1],ve[0].hw_frame->linesize[1],
            ve[1].hw_frame->data[0],ve[1].hw_frame->linesize[0], ve[1].hw_frame->data[1],ve[1].hw_frame->linesize[1],
            width_,height_,width_/2,height_,stream);
    }else if(ve.size()==4){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->rgba_to_nv12x4_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].hw_frame->data[0],ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1],ve[0].hw_frame->linesize[1],
            ve[1].hw_frame->data[0],ve[1].hw_frame->linesize[0], ve[1].hw_frame->data[1],ve[1].hw_frame->linesize[1],
            ve[2].hw_frame->data[0],ve[2].hw_frame->linesize[0], ve[2].hw_frame->data[1],ve[2].hw_frame->linesize[1],
            ve[3].hw_frame->data[0],ve[3].hw_frame->linesize[0], ve[3].hw_frame->data[1],ve[3].hw_frame->linesize[1],
            width_,height_,width_/2,height_/2,stream);
    }else if(ve.size()==8){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->rgba_to_nv12x8_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].hw_frame->data[0],ve[0].hw_frame->linesize[0], ve[0].hw_frame->data[1],ve[0].hw_frame->linesize[1],
            ve[1].hw_frame->data[0],ve[1].hw_frame->linesize[0], ve[1].hw_frame->data[1],ve[1].hw_frame->linesize[1],
            ve[2].hw_frame->data[0],ve[2].hw_frame->linesize[0], ve[2].hw_frame->data[1],ve[2].hw_frame->linesize[1],
            ve[3].hw_frame->data[0],ve[3].hw_frame->linesize[0], ve[3].hw_frame->data[1],ve[3].hw_frame->linesize[1],
            ve[4].hw_frame->data[0],ve[4].hw_frame->linesize[0], ve[4].hw_frame->data[1],ve[4].hw_frame->linesize[1],
            ve[5].hw_frame->data[0],ve[5].hw_frame->linesize[0], ve[5].hw_frame->data[1],ve[5].hw_frame->linesize[1],
            ve[6].hw_frame->data[0],ve[6].hw_frame->linesize[0], ve[6].hw_frame->data[1],ve[6].hw_frame->linesize[1],
            ve[7].hw_frame->data[0],ve[7].hw_frame->linesize[0], ve[7].hw_frame->data[1],ve[7].hw_frame->linesize[1],
            width_,height_,width_/4,height_/2,stream);
    }

    //本カーネル同期
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    // ------------------------
    // エンコード
    // ------------------------
    // 1. 全 encoder に frame を送る
    for (int i = 0; i < ve.size(); i++) {
        ve[i].hw_frame->pts = frame_index;
        avcodec_send_frame(ve[i].codec_ctx, ve[i].hw_frame);
    }

    // ---- mux（順序保証）----
    for (int i = 0; i < ve.size(); i++) {
        while (true) {
            int ret = avcodec_receive_packet(ve[i].codec_ctx, packet);

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(packet);
                break;
            }

            av_packet_rescale_ts(
                packet,
                ve[i].codec_ctx->time_base,
                ve[i].stream->time_base
                );

            packet->stream_index = ve[i].stream->index;

            av_interleaved_write_frame(fmt_ctx, packet);
            av_packet_unref(packet);
        }
    }

    // ========================
    // Audio encode（★ここ）
    // ========================
    encode_audio(Frame);
    frame_index+=1;
}

void save_encode::encode_audio(VideoFrame Frame)
{
    if (!audio_enc_ctx || !audio_fifo || !swr_enc) return;

    const int ch     = audio_enc_ctx->ch_layout.nb_channels;
    const int in_bps = 2; // S16

    for (const QByteArray& pcm : Frame.audio_pcm) {

        int in_samples = pcm.size() / (ch * in_bps);
        if (in_samples <= 0) continue;

        const uint8_t* in_data[1] = {
            reinterpret_cast<const uint8_t*>(pcm.constData())
        };

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

    // ---- AAC frame 単位で吐き出し ----
    const int fs = audio_enc_ctx->frame_size;

    while (av_audio_fifo_size(audio_fifo) >= fs) {

        AVFrame* frame = av_frame_alloc();
        frame->nb_samples = fs;
        frame->format = audio_enc_ctx->sample_fmt;
        frame->sample_rate = audio_enc_ctx->sample_rate;
        av_channel_layout_copy(&frame->ch_layout, &audio_enc_ctx->ch_layout);

        av_frame_get_buffer(frame, 0);

        // ★ PTS は「取り出したサンプル数」基準
        int read_samples = av_audio_fifo_read(audio_fifo,
                                              (void**)frame->data,
                                              fs);

        frame->pts = audio_pts;
        qDebug()<<"encode"<<frame->pts;
        audio_pts += 1024;   // ★ 必ず read_samples で進める

        avcodec_send_frame(audio_enc_ctx, frame);
        av_frame_free(&frame);


        AVPacket pkt;
        av_init_packet(&pkt);

        while (avcodec_receive_packet(audio_enc_ctx, &pkt) == 0) {
            av_packet_rescale_ts(
                &pkt,
                audio_enc_ctx->time_base,
                audio_stream->time_base
                );
            pkt.stream_index = audio_stream->index;
            av_interleaved_write_frame(fmt_ctx, &pkt);
            av_packet_unref(&pkt);
        }
    }
}
