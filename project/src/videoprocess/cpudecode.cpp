#include "src/videoprocess/cpudecode.h"

// ===============================
// mp4 CPUデコード
// ===============================

// ffmpeg初期化（CPU）
bool cpudecode::initialized_ffmpeg()
{
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    packet = av_packet_alloc();
    int ret;

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
    int video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        Error_String = "No video streams found";
        return false;
    }

    // ------------------------
    // bit depth 判定（CPU）
    // ------------------------
    AVCodecParameters* par = fmt_ctx->streams[video_stream_index]->codecpar;

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

    qDebug() << "bitdepth:" << VideoInfo.bitdepth;

    // ------------------------
    // デコーダ作成（CPU）
    // ------------------------
    {
        VideoDecorder dec;
        dec.stream_index = video_stream_index;

        // リングバッファ構築
        dec.hw_frames.resize(ringSize);
        for (int i = 0; i < ringSize; i++) {
            AVFrame* f = av_frame_alloc();
            if (!f) {
                Error_String = "av_frame_alloc failed";
                return false;
            }
            dec.hw_frames[i] = f;
        }

        AVCodecParameters* codecpar =
            fmt_ctx->streams[video_stream_index]->codecpar;

        const char* codec_name =
            avcodec_get_name(codecpar->codec_id);

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

        ret = avcodec_parameters_to_context(dec.codec_ctx, codecpar);
        if (ret < 0) {
            Error_String = QString("avcodec_parameters_to_context failed: %1")
            .arg(ffmpegErrStr(ret));
            return false;
        }

        // CPU デコード
        dec.codec_ctx->hw_device_ctx = nullptr;
        dec.codec_ctx->pkt_timebase =
            fmt_ctx->streams[video_stream_index]->time_base;

        // H.264 のみ並列デコード
        if (codecpar->codec_id == AV_CODEC_ID_H264) {
            dec.codec_ctx->thread_count = 0;          // auto
            dec.codec_ctx->thread_type  = FF_THREAD_FRAME;
        }

        ret = avcodec_open2(dec.codec_ctx, dec.decoder, nullptr);
        if (ret < 0) {
            Error_String = QString("avcodec_open2 failed (%1): %2")
            .arg(decoder_name)
                .arg(ffmpegErrStr(ret));
            return false;
        }

        vd.push_back(dec);
    }

    // ------------------------
    // 動画情報取得
    // ------------------------
    VideoInfo.fps = getFrameRate(fmt_ctx, vd[0].stream_index);

    double time_base_d =
        av_q2d(fmt_ctx->streams[vd[0].stream_index]->time_base);

    VideoInfo.pts_per_frame =
        1.0 / (VideoInfo.fps * time_base_d);

    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    if (!get_last_frame_pts()) {
        Error_String = "get_last_frame_pts failed";
        return false;
    }

    // ------------------------
    // デコードモード
    // ------------------------
    VideoInfo.decode_mode   = "Tiled Multi-Stream:stream_x1(tile:1)\n";
    VideoInfo.width_scale  = 1;
    VideoInfo.height_scale = 1;

    // ------------------------
    // CUDA メモリ確保（CPU decode + GPU upload）
    // ------------------------
    int bytesPerSample = (VideoInfo.bitdepth == 10) ? 2 : 1;

    size_t y_size{},uv_size{};
    y_size  = VideoInfo.width * VideoInfo.height * bytesPerSample;
    uv_size = (VideoInfo.width / 2) * (VideoInfo.height / 2) * bytesPerSample;

    cudaError_t err;

    err = cudaMallocPitch(&Frame.d_decode_rgba, &Frame.decode_pitch,
                          VideoInfo.width * 4,
                          VideoInfo.height);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch(d_rgba) failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }
    err = cudaMallocPitch(&Frame.d_encode_rgba,&Frame.encode_pitch,
                          VideoInfo.width * VideoInfo.width_scale * 4,
                          VideoInfo.height * VideoInfo.height_scale);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    err = cudaMallocPitch(&d_y, &pitch_y,
                          VideoInfo.width * bytesPerSample,
                          VideoInfo.height);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch(d_y) failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    err = cudaMallocPitch(&d_u, &pitch_u,
                          (VideoInfo.width / 2) * bytesPerSample,
                          VideoInfo.height / 2);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch(d_u) failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    err = cudaMallocPitch(&d_v, &pitch_v,
                          (VideoInfo.width / 2) * bytesPerSample,
                          VideoInfo.height / 2);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch(d_v) failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    // pinned 登録
    for (int i = 0; i < ringSize; i++) {
        cudaHostRegister(vd[0].hw_frames[i]->data[0], y_size,  cudaHostRegisterDefault);
        cudaHostRegister(vd[0].hw_frames[i]->data[1], uv_size, cudaHostRegisterDefault);
        cudaHostRegister(vd[0].hw_frames[i]->data[2], uv_size, cudaHostRegisterDefault);
    }

    // CUDA Stream
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        Error_String = QString("cudaStream failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    // CUDA Event
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        Error_String = QString("cudaEvent failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
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
        VideoInfo.audio = false;
    } else {
        audio_frame = av_frame_alloc();

        audio_decoder =
            avcodec_find_decoder(fmt_ctx->streams[audio_stream_index]->codecpar->codec_id);
        if (!audio_decoder) {
            Error_String = "Audio decoder not found";
            return false;
        }

        audio_ctx = avcodec_alloc_context3(audio_decoder);
        avcodec_parameters_to_context(
            audio_ctx,
            fmt_ctx->streams[audio_stream_index]->codecpar
            );

        ret = avcodec_open2(audio_ctx, audio_decoder, nullptr);
        if (ret < 0) {
            Error_String = QString("avcodec_open2(audio) failed: %1")
            .arg(ffmpegErrStr(ret));
            return false;
        }

        VideoInfo.in_sample_rate = audio_ctx->sample_rate;
        VideoInfo.in_format      = audio_ctx->sample_fmt;
        av_channel_layout_copy(&in_ch_layout, &audio_ctx->ch_layout);

        VideoInfo.out_sample_rate = audio_ctx->sample_rate;
        VideoInfo.out_format      = AV_SAMPLE_FMT_S16;
        av_channel_layout_default(&out_ch_layout, 2);

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

        QAudioFormat fmt;
        fmt.setSampleRate(VideoInfo.out_sample_rate);
        fmt.setChannelCount(2);
        fmt.setSampleFormat(QAudioFormat::Int16);

        audioSink   = new QAudioSink(fmt);
        audioOutput = audioSink->start();

        VideoInfo.audio = true;
    }

    //スライダー設定
    emit send_slider_info();

    // ------------------------
    // 再生初期化
    // ------------------------
    video_play_flag     = true;
    video_reverse_flag  = false;
    slider_No           = 0;

    return true;
}

//デコーダ設定
const char*cpudecode::selectDecoder(const char* codec_name) {
    const char*codec{};
    if (strcmp(codec_name, "h264") == 0) {
        codec="h264";
        qDebug()<<codec;
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc";
        qDebug()<<codec;
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1";
        qDebug()<<codec;
    }else{
        codec=nullptr;
    }
    VideoInfo.Codec=codec;
    return codec;
}

//フレームレートを取得する関数
double cpudecode::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
    if (video_stream_index < 0 || video_stream_index >= fmt_ctx->nb_streams) {
        qDebug() << "Invalid video stream index";
        return 0.0;
    }

    //映像ストリームの取得
    AVStream* video_stream = fmt_ctx->streams[video_stream_index];

    //avg_frame_rateを取得
    AVRational frame_rate = video_stream->avg_frame_rate;
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        // avg_frame_rate が0の時は r_frame_rate を使用
        frame_rate = video_stream->r_frame_rate;
    }

    //フレームレートを計算
    double frame_rate_value = static_cast<double>(frame_rate.num) / static_cast<double>(frame_rate.den);

    return frame_rate_value;
}

//テストでコード、最終フレームpts取得
bool cpudecode::get_last_frame_pts() {
    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t duration_pts = fmt_ctx->streams[vd[0].stream_index]->duration;
    if (duration_pts <= 0) {
        // durationが不明な場合はファイル末尾にシーク
        duration_pts = INT64_MAX;
    }

    // 後方シーク（できるだけ終端に近づく）
    if (av_seek_frame(fmt_ctx, vd[0].stream_index, duration_pts, AVSEEK_FLAG_BACKWARD) < 0) {
        Error_String = "seek failed";
        return false;
    }

    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t last_pts = AV_NOPTS_VALUE;
    bool frame_received = false;

    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);
        if (ret < 0) {
            // EOFに到達：デコーダに残りを流す
            avcodec_send_packet(vd[0].codec_ctx, nullptr);
            while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[0]) == 0) {
                last_pts = vd[0].hw_frames[0]->best_effort_timestamp;
                VideoInfo.width = vd[0].hw_frames[0]->width;
                VideoInfo.height = vd[0].hw_frames[0]->height;
                frame_received = true;
            }
            break;
        }

        if (packet->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[0]) == 0) {
                    last_pts = vd[0].hw_frames[0]->best_effort_timestamp;
                    VideoInfo.width = vd[0].hw_frames[0]->width;
                    VideoInfo.height = vd[0].hw_frames[0]->height;
                    frame_received = true;
                }
            }
        }
    }

    //packet解放
    av_packet_unref(packet);

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double time = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << time << "sec)";
        VideoInfo.max_framesNo = last_pts/VideoInfo.pts_per_frame;
        VideoInfo.start_range_framesNo = 0;
        VideoInfo.end_range_framesNo = VideoInfo.max_framesNo;
        Frame.FrameNo = VideoInfo.max_framesNo;

        //最大時間を計算 時:分:秒
        VideoInfo.max_hour = time / 3600;
        VideoInfo.max_minute = (int(time) % 3600) / 60;
        VideoInfo.max_second = fmod(time, 60.0);
    } else {
        Error_String = "No frame found at end.";
        return false;
    }

    return true;
}

//映像ストリーム取得
void cpudecode::get_decode_image(){
    //qDebug()<<"1フレーム読み込み"<<Frame.FrameNo;
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
                      slider_No * VideoInfo.pts_per_frame,
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

            if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                av_packet_unref(packet);
                break;
            }

            // 終端→先頭に戻ってループ
            if (Frame.FrameNo < VideoInfo.end_range_framesNo - 1) {
                avcodec_flush_buffers(vd[0].codec_ctx);
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                av_seek_frame(fmt_ctx, vd[0].stream_index,
                              (Frame.FrameNo + 1) * VideoInfo.pts_per_frame,
                              AVSEEK_FLAG_ANY);
            } else {
                emit decode_end();
                high_res_seek_frame(VideoInfo.start_range_framesNo,false);
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
            if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
                }
            }
        }
        av_packet_unref(packet);
    }

    gpu_upload();
    av_packet_unref(packet);
}

//オーディオ
void cpudecode::get_decode_audio()
{
    if (avcodec_send_packet(audio_ctx, packet) < 0)
        return;

    while (avcodec_receive_frame(audio_ctx, audio_frame) >= 0) {

        const int out_channels = out_ch_layout.nb_channels;
        const int bps = av_get_bytes_per_sample(VideoInfo.out_format);

        int in_rate = audio_frame->sample_rate > 0
                          ? audio_frame->sample_rate
                          : audio_ctx->sample_rate;

        int max_out_samples = av_rescale_rnd(
            swr_get_delay(swr, in_rate) + audio_frame->nb_samples,
            VideoInfo.out_sample_rate,
            in_rate,
            AV_ROUND_UP
            );


        QByteArray pcm_tmp;
        pcm_tmp.resize(max_out_samples * out_channels * bps);

        uint8_t* out_planes[] = {
            reinterpret_cast<uint8_t*>(pcm_tmp.data())
        };

        int out_samples = swr_convert(
            swr,
            out_planes,
            max_out_samples,
            (const uint8_t**)audio_frame->extended_data,
            audio_frame->nb_samples
            );

        if (out_samples <= 0)
            continue;

        pcm_tmp.resize(out_samples * out_channels * bps);
        VideoInfo.audio_channels = out_channels;

        // ★ここで確定コピーを作る
        QByteArray pcm=pcm_tmp;

        // 低遅延再生（同一スレッド）
        Frame.audio_pcm.push_back(QByteArray(pcm));
        Frame.audio_pts.push_back(audio_frame->pts);

        //低遅延音声再生
        if (encode_state == STATE_NOT_ENCODE) {
            if (g_AppSettings.audio_low_laytency_flag && audioOutput) {
                if (audioSink->bytesFree() >= pcm.size()) {

                    float volume = g_AppSettings.audio_volume / 100.0f;

                    int16_t* samples = reinterpret_cast<int16_t*>(pcm.data());
                    int sampleCount = pcm.size() / sizeof(int16_t);

                    for (int i = 0; i < sampleCount; i++) {
                        int32_t v = samples[i] * volume;

                        // クリップ防止
                        if (v > 32767) v = 32767;
                        if (v < -32768) v = -32768;

                        samples[i] = static_cast<int16_t>(v);
                    }

                    audioOutput->write(pcm);
                }
            }
        }

        // エンコードスレッドへ（別スレッド）
        emit send_audio(pcm);
    }
}

//GPUへアップロード
void cpudecode::gpu_upload(){
    int bytesPerSample = (VideoInfo.bitdepth == 10) ? 2 : 1;
    //GPUアップロード
    cudaMemcpy2D(d_y, pitch_y,
                 vd[0].hw_frames[ringNo]->data[0], vd[0].hw_frames[ringNo]->linesize[0],
                 VideoInfo.width * bytesPerSample,
                 VideoInfo.height,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(d_u, pitch_u,
                 vd[0].hw_frames[ringNo]->data[1], vd[0].hw_frames[ringNo]->linesize[1],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(d_v, pitch_v,
                 vd[0].hw_frames[ringNo]->data[2], vd[0].hw_frames[ringNo]->linesize[2],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(st);
    cudaEventRecord(ev, st);
    cudaEventSynchronize(ev);

    // yuv420p → RGBA
    if(VideoInfo.bitdepth == 8){
        CUDA_IMG_Proc->yuv420p_to_RGBA_8bit(
            Frame.d_decode_rgba,
            Frame.decode_pitch,
            d_y,
            pitch_y,
            d_u,
            pitch_u,
            d_v,
            pitch_v,
            VideoInfo.width,
            VideoInfo.height,
            st
            );
    }else if(VideoInfo.bitdepth == 10){
        int is_be = (vd[0].hw_frames[ringNo]->format == AV_PIX_FMT_YUV420P10BE);
        CUDA_IMG_Proc->yuv420p_to_RGBA_10bit(
            Frame.d_decode_rgba,
            Frame.decode_pitch,
            d_y,
            pitch_y,
            d_u,
            pitch_u,
            d_v,
            pitch_v,
            VideoInfo.width,
            VideoInfo.height,
            is_be,
            st
            );
    }

    //CUDAカーネル同期
    cudaEventRecord(ev, st);
    cudaEventSynchronize(ev);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(st);
    cudaEventRecord(ev, st);
    cudaEventSynchronize(ev);

    //フレーム番号取得
    if(seek_flag){
        Frame.FrameNo = vd[0].hw_frames[ringNo]->best_effort_timestamp / VideoInfo.pts_per_frame;
        slider_No = Frame.FrameNo;
        back1FrameNo = Frame.FrameNo-1;
    }else if(back1frame_flag||high_res_slider_flag){
        Frame.FrameNo = vd[0].hw_frames[ringNo]->best_effort_timestamp / VideoInfo.pts_per_frame;
        slider_No = Frame.FrameNo;
    }else{
        back1FrameNo = Frame.FrameNo;
        Frame.FrameNo = vd[0].hw_frames[ringNo]->best_effort_timestamp / VideoInfo.pts_per_frame;
        slider_No = Frame.FrameNo;
    }
    seek_flag = false;

    //時刻を算出
    double time = Frame.FrameNo/VideoInfo.fps;
    Frame.hour = time / 3600;
    Frame.minute = (int(time) % 3600) / 60;
    Frame.second = fmod(time, 60.0);

    emit send_decode_image(Frame,false,video_reverse_flag);
}

//高精度シーク
void cpudecode::high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag){
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
    if(heavy_UI_flag){
        //UIはfalse
        emit heavy_process_signal(false);
    }

    // ----------- シーク処理 -----------
    avcodec_flush_buffers(vd[0].codec_ctx);
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    av_seek_frame(fmt_ctx, vd[0].stream_index,
                  targetFrameNo * VideoInfo.pts_per_frame,
                  AVSEEK_FLAG_BACKWARD);

    // ----------- パケット読み込みループ -----------
    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }
    while(!thread_stop_flag){
        while (true) {
            int ret = av_read_frame(fmt_ctx, packet);

            // ---------- EOF ----------
            if (ret < 0) {
                avcodec_send_packet(vd[0].codec_ctx, nullptr);
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
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
                              seek_frame * VideoInfo.pts_per_frame,
                              AVSEEK_FLAG_ANY);

                av_packet_unref(packet);
                continue;
            }

            // ----------- VIDEO PACKET -----------
            if (packet->stream_index == vd[0].stream_index) {
                if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                    if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                        av_packet_unref(packet);
                        break;
                    }
                }
            }
        }

        //ターゲットフレームの番号かどうか判定
        back1FrameNo = FrameNo;
        FrameNo = vd[0].hw_frames[ringNo]->best_effort_timestamp / VideoInfo.pts_per_frame;

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo <= FrameNo){
            gpu_upload();
            break;
        }
        av_packet_unref(packet);
    }
}

