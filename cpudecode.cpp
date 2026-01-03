#include "cpudecode.h"

// ===============================
// mp4 CPUデコード
// ===============================

// ffmpeg初期化（CPU）
bool cpudecode::initialized_ffmpeg()
{
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

        dec.Frame = av_frame_alloc();
        if (!dec.Frame) {
            Error_String = "av_frame_alloc failed";
            return false;
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

    y_size  = VideoInfo.width * VideoInfo.height * bytesPerSample;
    uv_size = (VideoInfo.width / 2) * (VideoInfo.height / 2) * bytesPerSample;

    cudaError_t err;

    err = cudaMallocPitch(&d_rgba, &pitch_rgba,
                          VideoInfo.width * 4,
                          VideoInfo.height);
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch(d_rgba) failed: %1")
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
    cudaHostRegister(vd[0].Frame->data[0], y_size,  cudaHostRegisterDefault);
    cudaHostRegister(vd[0].Frame->data[1], uv_size, cudaHostRegisterDefault);
    cudaHostRegister(vd[0].Frame->data[2], uv_size, cudaHostRegisterDefault);

    // CUDA Stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        Error_String = QString("cudaStream failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    // CUDA Event
    cudaEventCreateWithFlags(&events, cudaEventDisableTiming);
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

        in_sample_rate = audio_ctx->sample_rate;
        in_format      = audio_ctx->sample_fmt;
        av_channel_layout_copy(&in_ch_layout, &audio_ctx->ch_layout);

        out_sample_rate = audio_ctx->sample_rate;
        out_format      = AV_SAMPLE_FMT_S16;
        av_channel_layout_default(&out_ch_layout, 2);

        if (swr) {
            swr_free(&swr);
        }

        ret = swr_alloc_set_opts2(
            &swr,
            &out_ch_layout,
            out_format,
            out_sample_rate,
            &in_ch_layout,
            in_format,
            in_sample_rate,
            0,
            nullptr
            );

        if (ret < 0 || swr_init(swr) < 0) {
            Error_String = "swr_init failed";
            return false;
        }

        QAudioFormat fmt;
        fmt.setSampleRate(out_sample_rate);
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
    const char*codec="";
    if (strcmp(codec_name, "h264") == 0) {
        codec="h264";
        qDebug()<<codec;
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc";
        qDebug()<<codec;
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1";
        qDebug()<<codec;
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
            while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                last_pts = vd[0].Frame->best_effort_timestamp;
                VideoInfo.width = vd[0].Frame->width;
                VideoInfo.height = vd[0].Frame->height;
                frame_received = true;
            }
            break;
        }

        if (packet->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                    last_pts = vd[0].Frame->best_effort_timestamp;
                    VideoInfo.width = vd[0].Frame->width;
                    VideoInfo.height = vd[0].Frame->height;
                    frame_received = true;
                }
            }
        }
    }

    //packet解放
    av_packet_unref(packet);

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double seconds = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << seconds << "sec)";
        VideoInfo.max_framesNo = fmt_ctx->streams[vd[0].stream_index]->duration/VideoInfo.pts_per_frame-1;
        VideoInfo.current_frameNo = VideoInfo.max_framesNo;
    } else {
        Error_String = "No frame found at end.";
        return false;
    }

    return true;
}

//映像ストリーム取得
void cpudecode::get_decode_image(){
    // ----------- シーク処理 -----------
    if (slider_No != VideoInfo.current_frameNo || video_reverse_flag) {
        if (video_reverse_flag) {
            slider_No--;
            if (slider_No < 0)
                slider_No = VideoInfo.max_framesNo;
        }

        avcodec_flush_buffers(vd[0].codec_ctx);
        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);

        av_seek_frame(fmt_ctx, vd[0].stream_index,
                      slider_No * VideoInfo.pts_per_frame,
                      AVSEEK_FLAG_BACKWARD);

        VideoInfo.current_frameNo = slider_No;
    }

    // ----------- パケット読み込みループ -----------
    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);
        // ---------- EOF ----------
        if (ret < 0) {
            avcodec_send_packet(vd[0].codec_ctx, nullptr);

            if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                gpu_upload();
                av_packet_unref(packet);
                break;
            }

            // 終端→先頭に戻ってループ
            uint64_t seek_frame;
            if (VideoInfo.current_frameNo < VideoInfo.max_framesNo - 1) {
                seek_frame = VideoInfo.current_frameNo + 1;
            } else {
                emit decode_end();
                seek_frame = 0;
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

        // ----------- AUDIO PACKET -----------
        if (packet->stream_index == audio_stream_index) {
            get_decode_audio();
            av_packet_unref(packet);
            continue;
        }

        // ----------- VIDEO PACKET -----------
        if (packet->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, packet) == 0) {
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                    gpu_upload();
                    av_packet_unref(packet);
                    break;
                }
            }
        }

        av_packet_unref(packet);
    }
}

//オーディオ
void cpudecode::get_decode_audio(){
    if (avcodec_send_packet(audio_ctx, packet) >= 0) {
        while (avcodec_receive_frame(audio_ctx, audio_frame) >= 0) {

            int out_channels = out_ch_layout.nb_channels;
            int bps         = av_get_bytes_per_sample(out_format);

            int max_out_samples = av_rescale_rnd(
                swr_get_delay(swr, in_sample_rate) + audio_frame->nb_samples,
                out_sample_rate,
                in_sample_rate,
                AV_ROUND_UP
                );

            QByteArray pcm;
            VideoInfo.audio_channels=out_channels;
            pcm.resize(max_out_samples * out_channels * bps);

            uint8_t* out_planes[1];
            out_planes[0] = (uint8_t*)pcm.data();

            int out_samples = swr_convert(
                swr,
                out_planes,
                max_out_samples,
                (const uint8_t**)audio_frame->extended_data,
                audio_frame->nb_samples
                );

            if (audio_buffer_size < out_samples * 4 * 2) {
                audio_buffer_size = out_samples * 4 * 2;
                audio_buffer = (uint8_t*)realloc(audio_buffer, audio_buffer_size);
            }

            int samples = swr_convert(
                swr,
                &audio_buffer,
                out_samples,
                (const uint8_t**)audio_frame->extended_data,
                audio_frame->nb_samples
                );

            int bytes = samples * 2 * av_get_bytes_per_sample(out_format);

            if(encode_state==STATE_NOT_ENCODE){
                if (audio_mode) {
                    if (audioOutput && bytes > 0){
                        audioOutput->write((char*)audio_buffer, bytes);
                    }
                }else{
                    if (out_samples > 0) {
                        pcm.resize(out_samples * out_channels * bps);
                        emit send_audio(pcm);
                    }
                }
            }
        }
    }
}

//GPUへアップロード
void cpudecode::gpu_upload(){
    int bytesPerSample = (VideoInfo.bitdepth == 10) ? 2 : 1;
    //GPUアップロード
    cudaMemcpy2D(d_y, pitch_y,
                 vd[0].Frame->data[0], vd[0].Frame->linesize[0],
                 VideoInfo.width * bytesPerSample,
                 VideoInfo.height,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(d_u, pitch_u,
                 vd[0].Frame->data[1], vd[0].Frame->linesize[1],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(d_v, pitch_v,
                 vd[0].Frame->data[2], vd[0].Frame->linesize[2],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    // yuv420p → RGBA
    if(VideoInfo.bitdepth == 8){
        CUDA_IMG_Proc->yuv420p_to_RGBA_8bit(
            d_rgba,
            pitch_rgba,
            d_y,
            pitch_y,
            d_u,
            pitch_u,
            d_v,
            pitch_v,
            VideoInfo.width,
            VideoInfo.height,
            stream
            );
    }else if(VideoInfo.bitdepth == 10){
        CUDA_IMG_Proc->yuv420p_to_RGBA_10bit(
            d_rgba,
            pitch_rgba,
            d_y,
            pitch_y,
            d_u,
            pitch_u,
            d_v,
            pitch_v,
            VideoInfo.width,
            VideoInfo.height,
            stream
            );
    }

    //CUDAカーネル同期
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    //フレーム番号取得
    VideoInfo.current_frameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;
    slider_No = VideoInfo.current_frameNo;

    emit send_decode_image(d_rgba, pitch_rgba, VideoInfo.current_frameNo);
}
