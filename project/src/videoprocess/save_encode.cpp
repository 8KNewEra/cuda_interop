#include "src/videoprocess/save_encode.h"
#include "qdebug.h"

save_encode::save_encode(int h,int w) {
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    width_=w;
    height_=h;
    frame_index = 0;
    int ret = 0;

    // ------------------------------------------------------------------
    // ★ ring と NVENC の async_depth の整合をとる
    //   NVENC は async_depth 段ぶん packet を溜めてから出し始めるので、
    //   ring_capacity <= async_depth だとメインが永久に待つ（デッドロック）。
    //   必ず ring > async_depth になるようにクランプする。
    // ------------------------------------------------------------------
    if (encodeRingSize < 3) {
        qWarning() << "[save_encode] encodeRingSize is too small:" << encodeRingSize
                   << "-> pipeline depth will be limited (8以上推奨)";
    }
    async_depth_ = std::max(1, std::min(4, encodeRingSize - 2));
    qDebug() << "[save_encode] ring =" << encodeRingSize
             << " async_depth =" << async_depth_;

    // ① FormatContext は1回だけ
    ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, encodeSettings.encode_path.toUtf8().constData());
    if (ret < 0 || !fmt_ctx) throw std::runtime_error("Failed to allocate format context");

    // ② Encoder / Stream を複数作る
    qDebug()<<encodeSettings.encode_tile;
    for (int i = 0; i < encodeSettings.encode_tile; i++) {
        ve.emplace_back(std::make_unique<VideoEncoder>());

        //CUDAデバイスコンテキスト
        QString gpuId = QString::number(encodeSettings.tile_gpu_map[i]);
        qDebug()<<gpuId;
        ret = av_hwdevice_ctx_create(&ve[i]->hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);
        if (ret < 0) throw std::runtime_error("Failed to create CUDA device");

        initialized_ffmpeg_hardware_context(i);   // ここで hw_frames.resize() される
        initialized_ffmpeg_codec_context(i,encodeSettings.encode_tile);

        // ★ stream 作成だけ
        ve[i]->stream = avformat_new_stream(fmt_ctx, nullptr);
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
    if(VideoInfo.audio){
        init_audio_encoder();
        audioRunning = true;
        audioThread =
            std::thread(
                &save_encode::audio_loop,
                this
                );
    }

    // ③ ファイルオープンは1回
    ret = avio_open(&fmt_ctx->pb, encodeSettings.encode_path.toUtf8().constData(), AVIO_FLAG_WRITE);
    if (ret < 0) throw std::runtime_error("Failed to open output file");

    // ④ ヘッダ書き込みは全 stream 作成後
    ret = avformat_write_header(fmt_ctx, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to write header");

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
    if (fmt_ctx) {
        if (fmt_ctx->pb) {
            avio_flush(fmt_ctx->pb);
        }

        int tret = av_write_trailer(fmt_ctx);
        qDebug() << "trailer ret =" << tret;

        if (fmt_ctx->pb) {
            avio_closep(&fmt_ctx->pb);
        }
        avformat_free_context(fmt_ctx);
        fmt_ctx = nullptr;
    }

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

    if (fmt_ctx && (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER))
        ve[i]->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    //ビットレート周りの設定
    AVDictionary* opts = nullptr;

    // cbr, vbr, cq 切り替え
    if (encodeSettings.rc_mode == "cq") {
        // --- CQ モード ---
        av_dict_set_int(&opts, "cq", encodeSettings.cq, 0);
    } else if (encodeSettings.rc_mode == "vbr") {
        // --- VBR モード ---
        ve[i]->codec_ctx->bit_rate = encodeSettings.target_bit_rate;
        ve[i]->codec_ctx->rc_max_rate = encodeSettings.max_bit_rate;
        ve[i]->codec_ctx->rc_buffer_size = encodeSettings.max_bit_rate;
    } else if (encodeSettings.rc_mode == "cbr") {
        // --- CBR モード ---
        ve[i]->codec_ctx->bit_rate = encodeSettings.target_bit_rate;
        ve[i]->codec_ctx->rc_max_rate = encodeSettings.max_bit_rate;
        ve[i]->codec_ctx->rc_buffer_size = encodeSettings.target_bit_rate;
    }

    // 共通オプション
    av_dict_set(&opts, "preset", encodeSettings.preset.toUtf8().constData(), 0);
    av_dict_set(&opts, "tune", encodeSettings.tune.toUtf8().constData(), 0);
    av_dict_set_int(&opts, "g", encodeSettings.gop_size, 0);
    av_dict_set_int(&opts, "bf", encodeSettings.b_frames, 0);
    av_dict_set(&opts, "rc-lookahead", "0", 0);
    av_dict_set(&opts, "zerolatency", "1", 0);

    // ★ 変更点: async_depth を上げて NVENC 内部でもパイプラインさせる
    //    （元は "1"。ring との整合はコンストラクタでクランプ済み）
    //    もし挙動を元に戻したい場合はここを 1 にするだけ。
    av_dict_set_int(&opts, "async_depth", async_depth_, 0);

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

//エンコード
void save_encode::encode(VideoFrame Frame)
{
    if (!Frame.audio_pcm.isEmpty()&&VideoInfo.audio)
    {
        AudioJob job;

        job.audio_pcm =
            Frame.audio_pcm;

        job.audio_pts =
            Frame.audio_pts;

        {
            std::lock_guard<std::mutex>
                lock(audioMutex);

            audioQueue.push(
                std::move(job)
                );
        }
        audioCV.notify_one();
    }

    encode_video(Frame);
}

//映像エンコード（メインスレッド側 = 変換と発行のみ）
void save_encode::encode_video(VideoFrame Frame)
{
    const int slot = encodeRingNo;

    // ==========================================================
    // 0) ★ 上書き防止バリア
    //    この slot を使っていた「前のフレーム」が全エンコーダで
    //    完全に終わる（= packet が出て NVENC が入力refを離す）まで待つ。
    //    ここを通過した時点で hw_frames[slot] は誰も参照していない。
    // ==========================================================
    for (auto& e : ve) {
        wait_slot_free(*e);
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
    // 2) CUDA NV12変換
    // ==========================================================
    if (ve.size() == 1) {
        CUDA_IMG_Proc->Flip_RGBA_to_NV12(
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch,
            ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            Frame.d_encode_rgba, Frame.encode_pitch,
            width_, height_,
            st
            );
    }
    else if (ve.size() == 2) {
        CUDA_IMG_Proc->rgba_to_nv12x2_flip_split(
            Frame.d_encode_rgba, Frame.encode_pitch,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch, ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            ve[1]->hw_frames[slot].d_y, ve[1]->hw_frames[slot].y_pitch, ve[1]->hw_frames[slot].d_uv, ve[1]->hw_frames[slot].uv_pitch,
            width_, height_,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            st
            );
    }
    else if (ve.size() == 4) {
        CUDA_IMG_Proc->rgba_to_nv12x4_flip_split(
            Frame.d_encode_rgba, Frame.encode_pitch,
            ve[0]->hw_frames[slot].d_y, ve[0]->hw_frames[slot].y_pitch, ve[0]->hw_frames[slot].d_uv, ve[0]->hw_frames[slot].uv_pitch,
            ve[1]->hw_frames[slot].d_y, ve[1]->hw_frames[slot].y_pitch, ve[1]->hw_frames[slot].d_uv, ve[1]->hw_frames[slot].uv_pitch,
            ve[2]->hw_frames[slot].d_y, ve[2]->hw_frames[slot].y_pitch, ve[2]->hw_frames[slot].d_uv, ve[2]->hw_frames[slot].uv_pitch,
            ve[3]->hw_frames[slot].d_y, ve[3]->hw_frames[slot].y_pitch, ve[3]->hw_frames[slot].d_uv, ve[3]->hw_frames[slot].uv_pitch,
            width_, height_,
            width_ / encodeSettings.width_tile,
            height_ / encodeSettings.height_tile,
            st
            );
    }
    else if (ve.size() == 8) {
        CUDA_IMG_Proc->rgba_to_nv12x8_flip_split(
            Frame.d_encode_rgba, Frame.encode_pitch,
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
            st
            );
    }
    else {
        qWarning() << "[save_encode] unsupported tile count:" << (int)ve.size();
    }

    // ==========================================================
    // 3) 変換完了待ち
    //    ★ ここで待つ理由は「呼び出し元の Frame.d_encode_rgba を
    //      encode() から戻った時点で再利用してよい」という従来の契約を
    //      壊さないため。転送とエンコードは以降すべて非同期。
    //      → convert(N+1) が copy(N) / encode(N) と重なる。
    // ==========================================================
    cudaEventRecord(ev, st);
    cudaEventSynchronize(ev);

    // ==========================================================
    // 4) 各GPUへ転送を「発行するだけ」（待たない）
    // ==========================================================
    for (int i = 0; i < (int)ve.size(); i++)
    {
        FrameSlot& fs = ve[i]->hw_frames[slot];

        if (encodeSettings.tile_gpu_map[i] == g_openglDeviceID)
        {
            // 変換結果が直接 hw_frame に入っている（3で同期済み）
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

        if (!sent) {
            // 投入できなかったフレームは packet が出てこないので、
            // ここで slot を返さないと ring が枯れてメインが止まる
            release_slot(enc);
            continue;
        }

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
                QMutexLocker locker(&muxMutex);
                int wret = av_interleaved_write_frame(fmt_ctx, pkt);
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
//   occupied = submitted - completed
//   occupied < ring_capacity のとき、次に書く slot(submitted % ring) は空。
// ==============================================================
void save_encode::wait_slot_free(VideoEncoder& enc)
{
    std::unique_lock<std::mutex> lk(enc.mtx);
    enc.slot_cv.wait(lk, [&]{
        return (enc.submitted - enc.completed) < (uint64_t)enc.ring_capacity;
    });
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
        enc.submitted++;
    }
    enc.job_cv.notify_one();
}

void save_encode::release_slot(VideoEncoder& enc)
{
    {
        std::lock_guard<std::mutex> lk(enc.mtx);
        if (enc.completed < enc.submitted) enc.completed++;
    }
    enc.slot_cv.notify_all();
}

//音声エンコード
void save_encode::encode_audio(AudioJob Frame)
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
