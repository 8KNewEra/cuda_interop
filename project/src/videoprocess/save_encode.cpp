#include "src/videoprocess/save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    width_=w;
    height_=h;
    frame_index = 0;
    int ret = 0;
    packet = av_packet_alloc();

    // ① FormatContext は1回だけ
    ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, encodeSettings.encode_path.toUtf8().constData());
    if (ret < 0 || !fmt_ctx) throw std::runtime_error("Failed to allocate format context");

    // ② Encoder / Stream を複数作る
    qDebug()<<encodeSettings.encode_tile;
    for (int i = 0; i < encodeSettings.encode_tile; i++) {
        ve.emplace_back();

        //CUDAデバイスコンテキスト
        QString gpuId = QString::number(encodeSettings.tile_gpu_map[i]);
        qDebug()<<gpuId;
        ret = av_hwdevice_ctx_create(&ve[i].hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
        if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

        initialized_ffmpeg_hardware_context(i);
        initialized_ffmpeg_codec_context(i,encodeSettings.encode_tile);

        // ★ stream 作成だけ
        ve[i].stream = avformat_new_stream(fmt_ctx, nullptr);
        if (!ve[i].stream) throw std::runtime_error("Failed to create stream");

        ve[i].stream->time_base = ve[i].codec_ctx->time_base;
        ve[i].stream->avg_frame_rate = ve[i].codec_ctx->framerate;
        ve[i].stream->r_frame_rate   = ve[i].codec_ctx->framerate;
        ret = avcodec_parameters_from_context(ve[i].stream->codecpar, ve[i].codec_ctx);
        if (ret < 0) throw std::runtime_error("Failed to copy codec parameters");

        this->ve[i].stream = ve[i].stream;

        //GPU転送用のメモリを確保
        if(encodeSettings.tile_gpu_map[i] != g_openglDeviceID){
            cudaMallocPitch(
                &ve[i].d_y,
                &ve[i].y_pitch,
                width_/encodeSettings.width_tile,
                height_/encodeSettings.height_tile
                );
            cudaMallocPitch(
                &ve[i].d_uv,
                &ve[i].uv_pitch,
                (width_/encodeSettings.width_tile),
                (height_/encodeSettings.height_tile)/2
                );
        }

        //Stream作成
        cudaStreamCreateWithFlags(
            &ve[i].st,
            cudaStreamNonBlocking
            );

        //event作成
        cudaEventCreateWithFlags(
            &ve[i].ev,
            cudaEventDisableTiming
            );
    }

    if(VideoInfo.audio)
        init_audio_encoder();

    // ③ ファイルオープンは1回
    ret = avio_open(&fmt_ctx->pb, encodeSettings.encode_path.toUtf8().constData(), AVIO_FLAG_WRITE);
    if (ret < 0) throw std::runtime_error("Failed to open output file");

    qDebug() << "FINAL enc_tb="
             << ve[0].codec_ctx->time_base.num << "/"
             << ve[0].codec_ctx->time_base.den;
    qDebug() << "FINAL st_tb="
             << ve[0].stream->time_base.num << "/"
             << ve[0].stream->time_base.den;

    // ④ ヘッダ書き込みは全 stream 作成後
    ret = avformat_write_header(fmt_ctx, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to write header");

    qDebug() << "FINAL enc_tb="
             << ve[0].codec_ctx->time_base.num << "/"
             << ve[0].codec_ctx->time_base.den;
    qDebug() << "FINAL st_tb="
             << ve[0].stream->time_base.num << "/"
             << ve[0].stream->time_base.den;

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
}

save_encode::~save_encode() {
    // ==========================
    // GPU同期
    // ==========================
    for (int i = 0; i < ve.size(); i++) {
        if (ve[i].st) {
            cudaStreamSynchronize(ve[i].st);
        }
    }

    // ==========================
    // Video flush（NVENC）
    // ==========================
    // まず全 encoder に NULL frame を送る
    for (int i = 0; i < ve.size(); i++) {
        int ret = avcodec_send_frame(ve[i].codec_ctx, nullptr);
        if (ret < 0) {
            qDebug() << "send NULL frame error:" << ret;
        }
    }

    // drain を1回まとめて回す関数
    auto drain_video_once = [&]() -> bool {
        bool got_any = false;

        for (int i = 0; i < ve.size(); i++) {
            while (true) {
                int ret = avcodec_receive_packet(ve[i].codec_ctx, packet);

                if (ret == 0) {
                    got_any = true;

                    av_packet_rescale_ts(packet,
                                         ve[i].codec_ctx->time_base,
                                         ve[i].stream->time_base);

                    packet->stream_index = ve[i].stream->index;

                    int wret = av_interleaved_write_frame(fmt_ctx, packet);
                    if (wret < 0) {
                        char err[256];
                        av_strerror(wret, err, sizeof(err));
                        qDebug() << "write_frame error:" << wret << err;
                    }

                    av_packet_unref(packet);
                    continue;
                }

                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_packet_unref(packet);
                    break;
                }

                qDebug() << "flush receive error:" << ret;
                av_packet_unref(packet);
                break;
            }
        }

        return got_any;
    };

    // まず普通に drain しきる
    while (drain_video_once()) {}

    // ★ 最後だけ追加で100ms粘る（NVENC遅延対策）
    for (int t = 0; t < 100; t++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // もし何も出てこなければ終了
        if (!drain_video_once()) {
            break;
        }
    }

    // ==========================
    // Audio flush（AAC）
    // ==========================
    if (audio_enc_ctx && audio_fifo) {
        const int fs = audio_enc_ctx->frame_size;
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

    // ==========================
    // ファイルの終了処理
    // ==========================
    if (fmt_ctx && fmt_ctx->pb) {
        avio_flush(fmt_ctx->pb);
    }

    int tret = av_write_trailer(fmt_ctx);
    qDebug() << "trailer ret =" << tret;

    if (fmt_ctx && fmt_ctx->pb) {
        avio_closep(&fmt_ctx->pb);
    }
    avformat_free_context(fmt_ctx);
    fmt_ctx = nullptr;

    // ---- 各メモリ解放 ----
    for(int i=0;i<ve.size();i++){
        //ハードウェアフレームを解放
        for (AVFrame* &f : ve[i].hw_frames) {
            if (f) {
                av_frame_unref(f);
                av_frame_free(&f);
                f = nullptr;
            }
        }
        ve[i].hw_frames.clear();
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
        //ハードウェアデバイスコンテキストを解放
        if (ve[i].hw_device_ctx) {
            av_buffer_unref(&ve[i].hw_device_ctx);
            ve[i].hw_device_ctx = nullptr;
        }
        //メモリ開放
        if(encodeSettings.tile_gpu_map[i] != g_openglDeviceID){
            if (ve[i].d_y) {
                cudaFree(ve[i].d_y);
                ve[i].d_y = nullptr;
            }
            if (ve[i].d_uv) {
                cudaFree(ve[i].d_uv);
                ve[i].d_uv = nullptr;
            }
        }
        //Stream削除
        if(ve[i].st){
            cudaStreamSynchronize(ve[i].st);
            cudaStreamDestroy(ve[i].st);
            ve[i].st=nullptr;
        }
        //event削除
        if (ve[i].ev) {
            cudaEventDestroy(ve[i].ev);
            ve[i].ev = nullptr;
        }
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
    ve[i].codec_ctx->width = width_/encodeSettings.width_tile;
    ve[i].codec_ctx->height = height_/encodeSettings.height_tile;
    ve[i].codec_ctx->pix_fmt = hw_pix_fmt;

    //フレームレート設定
    AVRational fps = fps_to_rational(encodeSettings.save_fps);
    ve[i].codec_ctx->framerate = fps;
    ve[i].codec_ctx->time_base = av_inv_q(fps);

    //初期化された hw_* を参照
    ve[i].codec_ctx->hw_device_ctx = av_buffer_ref(ve[i].hw_device_ctx);
    ve[i].codec_ctx->hw_frames_ctx = av_buffer_ref(ve[i].hw_frames_ctx);

    //ビットレート周りの設定
    AVDictionary* opts = nullptr;

    // cbr, vbr, cq 切り替え
    if (encodeSettings.rc_mode == "cq") {
        // --- CQ モード ---
        av_dict_set_int(&opts, "cq", encodeSettings.cq, 0);
    } else if (encodeSettings.rc_mode == "vbr") {
        // --- VBR モード ---
        ve[i].codec_ctx->bit_rate = encodeSettings.target_bit_rate;
        ve[i].codec_ctx->rc_max_rate = encodeSettings.max_bit_rate;
        ve[i].codec_ctx->rc_buffer_size = encodeSettings.max_bit_rate;
    } else if (encodeSettings.rc_mode == "cbr") {
        // --- CBR モード ---
        ve[i].codec_ctx->bit_rate = encodeSettings.target_bit_rate;
        ve[i].codec_ctx->rc_max_rate = encodeSettings.max_bit_rate;
        ve[i].codec_ctx->rc_buffer_size = encodeSettings.target_bit_rate;
    }

    // 共通オプション
    av_dict_set(&opts, "preset", encodeSettings.preset.toUtf8().constData(), 0);
    av_dict_set(&opts, "tune", encodeSettings.tune.toUtf8().constData(), 0);
    av_dict_set_int(&opts, "g", encodeSettings.gop_size, 0);
    av_dict_set_int(&opts, "bf", encodeSettings.b_frames, 0);
    av_dict_set(&opts, "rc-lookahead", "0", 0);
    av_dict_set(&opts, "zerolatency", "1", 0);
    av_dict_set(&opts, "async_depth", "1", 0);

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

    int ret = avcodec_open2(ve[i].codec_ctx, codec, &opts);
    if (ret < 0) throw std::runtime_error("Failed to open codec");
    av_dict_free(&opts);
}

AVRational save_encode::fps_to_rational(double fps)
{
    // よくあるNTSC系は固定で返す（誤差吸収）
    if (fabs(fps - 59.94) < 0.01) return AVRational{60000, 1001};
    if (fabs(fps - 29.97) < 0.01) return AVRational{30000, 1001};
    if (fabs(fps - 23.976) < 0.01) return AVRational{24000, 1001};

    // それ以外は整数fpsに丸める（例: 60.0, 30.0）
    int fps_i = (int)llround(fps);
    if (fps_i <= 0) fps_i = 30;

    return AVRational{fps_i, 1};
}

//CUDAデバイスコンテキスト初期化
void save_encode::initialized_ffmpeg_hardware_context(int i)
{
    int ret = 0;

    // ---- hw_frames_ctx ----
    ve[i].hw_frames_ctx = av_hwframe_ctx_alloc(ve[i].hw_device_ctx);
    if (!ve[i].hw_frames_ctx) {
        throw std::runtime_error("Failed to allocate hw_frames_ctx");
    }

    AVHWFramesContext* frames_ctx =
        (AVHWFramesContext*)(ve[i].hw_frames_ctx->data);

    frames_ctx->format    = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width     = width_  / encodeSettings.width_tile;
    frames_ctx->height    = height_ / encodeSettings.height_tile;
    frames_ctx->initial_pool_size = ringSize + 4;
    ret = av_hwframe_ctx_init(ve[i].hw_frames_ctx);
    if (ret < 0) {
        throw std::runtime_error("Failed to init frames_ctx");
    }

    // hw_frameリング確保
    ve[i].hw_frames.resize(ringSize);

    // リングバッファ構築
    for (int k = 0; k < ringSize; k++) {
        AVFrame* f = av_frame_alloc();
        if (!f) throw std::runtime_error("av_frame_alloc failed");

        f->format = AV_PIX_FMT_CUDA;
        f->width  = frames_ctx->width;
        f->height = frames_ctx->height;

        ret = av_hwframe_get_buffer(ve[i].hw_frames_ctx, f, 0);
        if (ret < 0) {
            av_frame_free(&f);
            throw std::runtime_error("Failed to alloc hw_frame ring buffer");
        }
        ve[i].hw_frames[k] = f;
    }
}

//オーディオエンコーダー初期化
void save_encode::init_audio_encoder()
{
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec)
        throw std::runtime_error("AAC encoder not found");

    audio_enc_ctx = avcodec_alloc_context3(codec);
    audio_enc_ctx->sample_rate = VideoInfo.in_sample_rate;
    audio_enc_ctx->bit_rate = 192000;
    audio_enc_ctx->time_base = AVRational{1, audio_enc_ctx->sample_rate};

    av_channel_layout_default(
        &audio_enc_ctx->ch_layout,
        VideoInfo.audio_channels > 0 ? VideoInfo.audio_channels : 2
        );

    audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(audio_enc_ctx, codec, nullptr) < 0)
        throw std::runtime_error("avcodec_open2(audio) failed");

    if (audio_enc_ctx->frame_size <= 0)
        throw std::runtime_error("Invalid AAC frame_size");

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

//映像エンコード
void save_encode::encode(VideoFrame Frame){
    //gpu0のフレームはそのままhw_frameに書き込み
    for(int i = 0; i < ve.size(); i++)
    {
        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID)
        {
            AVFrame* out = ve[i].hw_frames[ve[i].ringNo];
            ve[i].d_y     = out->data[0];
            ve[i].y_pitch = out->linesize[0];
            ve[i].d_uv    = out->data[1];
            ve[i].uv_pitch= out->linesize[1];
        }
    }

    // ------------------------
    // CUDA NV12変換
    // ------------------------
    //本カーネル
    if(ve.size()==1){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->Flip_RGBA_to_NV12(ve[0].d_y,ve[0].y_pitch, ve[0].d_uv,ve[0].uv_pitch,Frame.d_encode_rgba, Frame.encode_pitch,width_,height_,st);
    }else if(ve.size()==2){
        CUDA_IMG_Proc->rgba_to_nv12x2_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].d_y,ve[0].y_pitch, ve[0].d_uv,ve[0].uv_pitch,
            ve[1].d_y,ve[1].y_pitch, ve[1].d_uv,ve[1].uv_pitch,
            width_,height_,
            width_/encodeSettings.width_tile,
            height_/encodeSettings.height_tile,
            st);
    }else if(ve.size()==4){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->rgba_to_nv12x4_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].d_y,ve[0].y_pitch, ve[0].d_uv,ve[0].uv_pitch,
            ve[1].d_y,ve[1].y_pitch, ve[1].d_uv,ve[1].uv_pitch,
            ve[2].d_y,ve[2].y_pitch, ve[2].d_uv,ve[2].uv_pitch,
            ve[3].d_y,ve[3].y_pitch, ve[3].d_uv,ve[3].uv_pitch,
            width_,height_,
            width_/encodeSettings.width_tile,
            height_/encodeSettings.height_tile,
            st);
    }else if(ve.size()==8){
        //RGBAをNV12に変換してffmpegへ転送
        CUDA_IMG_Proc->rgba_to_nv12x8_flip_split(
            Frame.d_encode_rgba,Frame.encode_pitch,
            ve[0].d_y,ve[0].y_pitch, ve[0].d_uv,ve[0].uv_pitch,
            ve[1].d_y,ve[1].y_pitch, ve[1].d_uv,ve[1].uv_pitch,
            ve[2].d_y,ve[2].y_pitch, ve[2].d_uv,ve[2].uv_pitch,
            ve[3].d_y,ve[3].y_pitch, ve[3].d_uv,ve[3].uv_pitch,
            ve[4].d_y,ve[4].y_pitch, ve[4].d_uv,ve[4].uv_pitch,
            ve[5].d_y,ve[5].y_pitch, ve[5].d_uv,ve[5].uv_pitch,
            ve[6].d_y,ve[6].y_pitch, ve[6].d_uv,ve[6].uv_pitch,
            ve[7].d_y,ve[7].y_pitch, ve[7].d_uv,ve[7].uv_pitch,
            width_,height_,
            width_/encodeSettings.width_tile,
            height_/encodeSettings.height_tile,
            st);
    }
    //変換完了のイベントを待つだけにする
    cudaEventRecord(ev, st);

    //各GPUへ転送
    for(int i = 0; i < ve.size(); i++)
    {
        cudaStreamWaitEvent(ve[i].st, ev, 0);

        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID){
            cudaEventRecord(ve[i].ev, ve[i].st);
            continue; // primaryはコピー不要
        }

        AVFrame* out = ve[i].hw_frames[ve[i].ringNo];
        cudaMemcpy2DAsync(
            out->data[0], out->linesize[0],
            ve[i].d_y, ve[i].y_pitch,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            cudaMemcpyDeviceToDevice,
            ve[i].st
            );

        cudaMemcpy2DAsync(
            out->data[1], out->linesize[1],
            ve[i].d_uv, ve[i].uv_pitch,
            width_ / encodeSettings.width_tile,
            (height_ / encodeSettings.height_tile) / 2,
            cudaMemcpyDeviceToDevice,
            ve[i].st
            );

        cudaEventRecord(ve[i].ev, ve[i].st);
    }
    // 全タイル完了待ち
    for(int i = 0; i < ve.size(); i++)
    {
        cudaEventSynchronize(ve[i].ev);
    }

    // ------------------------
    // エンコード
    // ------------------------
    // 1. 全 encoder に frame を送る
    for (int i = 0; i < ve.size(); i++) {
        ve[i].hw_frames[ve[i].ringNo]->pts = frame_index;
        avcodec_send_frame(ve[i].codec_ctx, ve[i].hw_frames[ve[i].ringNo]);
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

//音声エンコード
void save_encode::encode_audio(VideoFrame Frame)
{
    if (!audio_enc_ctx || !audio_fifo || !swr_enc) return;

    const int ch     = audio_enc_ctx->ch_layout.nb_channels;
    const int in_bps = av_get_bytes_per_sample(VideoInfo.out_format);// S16

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
        int read_samples = av_audio_fifo_read(audio_fifo,(void**)frame->data,fs);
        frame->pts = audio_pts;
        audio_pts += (int)(((double)read_samples)*(VideoInfo.fps/encodeSettings.save_fps));   // ★ 必ず read_samples で進める

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
