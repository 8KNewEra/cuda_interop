#include "nvgpudecode.h"

// ===============================
// mp4 NVIDIA向けGPUデコード
// ===============================

//ffmpeg初期化
void nvgpudecode::initialized_ffmpeg(){
    int ret;

    // CUDA デバイスコンテキスト作成
    QString gpuId = QString::number(g_cudaDeviceID);   // GPU1 を使う例
    qDebug()<<gpuId;
    ret = av_hwdevice_ctx_create(&hw_device_ctx,AV_HWDEVICE_TYPE_CUDA,gpuId.toUtf8().data(),nullptr,0);

    if (ret < 0) {
        QString error = "Failed to create CUDA device ctx on GPU " + gpuId + ":" + ffmpegErrStr(ret);
        emit decode_error(error);
        return;
    }

    // ファイルを開く
    ret = avformat_open_input(&fmt_ctx, input_filename, nullptr, nullptr);
    if (ret < 0) {
        QString error="Could not open input file:" + ffmpegErrStr(ret);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    //動画内ストリーム検索
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        QString error="Failed to retrieve input stream information:"+ ffmpegErrStr(ret);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    //映像ストリーム検索
    std::vector<int> video_stream_indices;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_indices.push_back(i);
        }
    }

    if (video_stream_indices.empty()) {
        emit decode_error("No video streams found");
        return;
    }

    //ビット深度取得
    AVCodecParameters* par = fmt_ctx->streams[0]->codecpar;
    if (par->codec_id == AV_CODEC_ID_HEVC||par->codec_id == AV_CODEC_ID_H264) {
        if (par->format == AV_PIX_FMT_YUV420P10 ||
            par->format == AV_PIX_FMT_P010) {
            VideoInfo.bitdepth = 10;
        }else{
            VideoInfo.bitdepth = 8;
        }
    }

    qDebug()<<VideoInfo.bitdepth;

    // ------------------------
    // デコーダ作成（GPU）
    // ------------------------
    //デコーダ作成
    for (int i = 0; i < video_stream_indices.size(); i++) {
        int stream_index = video_stream_indices[i];

        vd.emplace_back();
        auto& dec = vd.back();

        dec.stream_index = stream_index;
        dec.Frame = av_frame_alloc();

        const char* codec_name =
            avcodec_get_name(fmt_ctx->streams[stream_index]->codecpar->codec_id);

        const char* decoder_name = selectDecoder(codec_name);
        if (!decoder_name) {
            emit decode_error("Unsupported codec");
            return;
        }

        dec.decoder = avcodec_find_decoder_by_name(decoder_name);
        if (!dec.decoder) {
            emit decode_error("Decoder not found");
            return;
        }

        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "extra_hw_frames", "0", 0);

        dec.codec_ctx = avcodec_alloc_context3(dec.decoder);
        dec.codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

        ret = avcodec_parameters_to_context(
            dec.codec_ctx,
            fmt_ctx->streams[stream_index]->codecpar
            );
        if (ret < 0) {
            emit decode_error("Failed to copy codec parameters");
            return;
        }

        ret = avcodec_open2(dec.codec_ctx, dec.decoder, &opts);
        av_dict_free(&opts);

        if (ret < 0) {
            emit decode_error("Codec open error");
            return;
        }
    }

    // ------------------------
    // 動画情報取得
    // ------------------------
    //フレームレートを取得
    VideoInfo.fps = getFrameRate(fmt_ctx, vd[0].stream_index);

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[vd[0].stream_index]->time_base);
    VideoInfo.pts_per_frame = 1.0 / (VideoInfo.fps * time_base_d);
    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    //最終フレームptsを取得
    get_last_frame_pts();

    //デコードモード
    if(vd.size()==1){
        VideoInfo.decode_mode = "Decode Mode:Non split(tile:1)\n";
        VideoInfo.width_scale=1;
        VideoInfo.height_scale=1;
    }else if(vd.size()==2){
        VideoInfo.decode_mode = "Decode Mode:split_x2(tile:2×1)\n";
        VideoInfo.width_scale=2;
        VideoInfo.height_scale=1;
    }else if(vd.size()==4){
        VideoInfo.decode_mode = "Decode Mode:split_x4(tile:2×2)\n";
        VideoInfo.width_scale=2;
        VideoInfo.height_scale=2;
    }

    // ------------------------
    // CUDA周り設定
    // ------------------------
    //CUDA周りメモリ確保
    cudaError_t err = cudaMallocPitch(
        &d_rgba,
        &pitch_rgba,
        VideoInfo.width * VideoInfo.width_scale * 4,
        VideoInfo.height * VideoInfo.height_scale
        );

    if (err != cudaSuccess) {
        qDebug() << "cudaMallocPitch failed"
                 << cudaGetErrorString(err);
    }

    //CUDA Stream作成
    cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking
        );

    //CUDA event 作成
    cudaEventCreateWithFlags(
        &events,
        cudaEventDisableTiming
        );

    // -----------------------------
    // 音声ストリームの探索
    // -----------------------------
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    if (audio_stream_index == -1) {
        qDebug() << "Audio stream not found";
        VideoInfo.audio=false;
    } else {
        // --------- Audio Decoder ----------
        audio_decoder = avcodec_find_decoder(fmt_ctx->streams[audio_stream_index]->codecpar->codec_id);
        audio_ctx = avcodec_alloc_context3(audio_decoder);

        avcodec_parameters_to_context(audio_ctx, fmt_ctx->streams[audio_stream_index]->codecpar);

        if (avcodec_open2(audio_ctx, audio_decoder, nullptr) < 0) {
            qDebug() << "Failed to open audio decoder";
            audio_stream_index = -1;
            return;
        }
        qDebug() << "Audio decoder opened";

        // 入力情報を member に保存
        in_sample_rate = audio_ctx->sample_rate;
        in_format      = audio_ctx->sample_fmt;
        av_channel_layout_copy(&in_ch_layout, &audio_ctx->ch_layout);

        // 出力情報（固定）
        out_sample_rate = audio_ctx->sample_rate;  // 入力と同じでOK
        out_format      = AV_SAMPLE_FMT_S16;

        // 出力レイアウト（ステレオ）→ メンバ変数に設定
        av_channel_layout_default(&out_ch_layout, 2);

        // SwrContext 作成 (メンバ変数 swr を利用)
        if (swr) {
            swr_free(&swr);
        }

        int ret = swr_alloc_set_opts2(
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
            qDebug() << "Failed to init swr";
            return;
        }

        // ---------- QAudioSink ----------
        QAudioFormat fmt;
        fmt.setSampleRate(out_sample_rate);
        fmt.setChannelCount(2);
        fmt.setSampleFormat(QAudioFormat::Int16);

        audioSink = new QAudioSink(fmt);
        audioOutput = audioSink->start();

        qDebug() << "Audio ready";
        VideoInfo.audio=true;
    }

    //スライダー設定
    emit send_slider_info();

    //再生
    video_play_flag = true;
    video_reverse_flag = false;
    slider_No=0;
}

// デコーダ設定
const char*nvgpudecode::selectDecoder(const char* codec_name) {
    const char*codec="";
    if (strcmp(codec_name, "h264") == 0) {
        codec="h264_cuvid";
        qDebug()<<codec;
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc_cuvid";
        qDebug()<<codec;
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1_cuvid";
        qDebug()<<codec;
    }
    VideoInfo.Codec=codec;
    return codec;
}

//フレームレートを取得する関数
double nvgpudecode::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
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

//最終フレームpts取得
void nvgpudecode::get_last_frame_pts() {
    AVPacket* pkt = av_packet_alloc();
    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t duration_pts = fmt_ctx->streams[vd[0].stream_index]->duration;
    if (duration_pts <= 0) {
        // durationが不明な場合はファイル末尾にシーク
        duration_pts = INT64_MAX;
    }

    // 後方シーク（できるだけ終端に近づく）
    if (av_seek_frame(fmt_ctx, vd[0].stream_index, duration_pts, AVSEEK_FLAG_BACKWARD) < 0) {
        qDebug() << "seek failed";
        return;
    }

    avcodec_flush_buffers(vd[0].codec_ctx);

    int64_t last_pts = AV_NOPTS_VALUE;
    bool frame_received = false;

    while (true) {
        int ret = av_read_frame(fmt_ctx, pkt);
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

        if (pkt->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                    last_pts = vd[0].Frame->best_effort_timestamp;
                    VideoInfo.width = vd[0].Frame->width;
                    VideoInfo.height = vd[0].Frame->height;
                    frame_received = true;
                }
            }
        }
        av_packet_unref(pkt);
    }

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double seconds = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << seconds << "sec)";
        VideoInfo.max_framesNo = fmt_ctx->streams[vd[0].stream_index]->duration/VideoInfo.pts_per_frame-1;
        VideoInfo.current_frameNo = VideoInfo.max_framesNo;
    } else {
        qDebug() << "No frame found at end.";
    }
}

//複数ストリームフレーム取得
void nvgpudecode::get_decode_image() {
    AVPacket* pkt = av_packet_alloc();
    std::vector<bool> got_frame(vd.size(), false);
    int got_count = 0;

    //----------- シーク処理 -----------
    if (slider_No != VideoInfo.current_frameNo || video_reverse_flag) {
        if (video_reverse_flag) {
            slider_No--;
            if (slider_No < 0)
                slider_No = VideoInfo.max_framesNo;
        }

        // ビデオ/オーディオの両方 flush
        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);

        for (auto& dec : vd) {
            avcodec_flush_buffers(dec.codec_ctx);
        }

        av_seek_frame(fmt_ctx, 0,
                      slider_No * VideoInfo.pts_per_frame,
                      AVSEEK_FLAG_BACKWARD);

        VideoInfo.current_frameNo = slider_No;
    }

    while (got_count < vd.size()) {
        int ret = av_read_frame(fmt_ctx, pkt);
        // ---------- EOF ----------
        if (ret < 0) {
            for (int i = 0; i < vd.size(); i++) {
                // 映像側フラッシュ
                avcodec_send_packet(vd[i].codec_ctx, nullptr);

                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                    got_frame[i] = true;
                    got_count++;
                }
            }

            //終端→先頭に戻ってループ
            if (VideoInfo.current_frameNo >= VideoInfo.max_framesNo) {
                emit decode_end();

                //複数ストリームは0でシーク
                for (auto& dec : vd) { avcodec_flush_buffers(dec.codec_ctx); }
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                av_seek_frame(fmt_ctx, 0,
                              0 * VideoInfo.pts_per_frame,
                              AVSEEK_FLAG_ANY);

                continue;
            }

            if(got_count==vd.size()){
                CUDA_RGBA_to_merge();
                av_packet_unref(pkt);
            }
        }

        // ----------- AUDIO PACKET -----------
        if (pkt->stream_index == audio_stream_index) {
            get_decode_audio(pkt);
            continue;  // → 映像パケットを探しに行く
        }

        // ----------- VIDEO PACKET -----------
        for (int i = 0; i < vd.size(); i++) {
            if (pkt->stream_index == vd[i].stream_index&& !got_frame[i]) {
                if (avcodec_send_packet(vd[i].codec_ctx, pkt) < 0) {
                    continue;
                }

                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                    got_frame[i] = true;
                    got_count++;
                }
            }
        }
    }

    // CUDAの処理,NVDEC完了待ち
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    CUDA_RGBA_to_merge();

    // パケット開放
    av_packet_unref(pkt);
}

//オーディオ
void nvgpudecode::get_decode_audio(AVPacket* pkt){
    if (avcodec_send_packet(audio_ctx, pkt) >= 0) {
        AVFrame* af = av_frame_alloc();
        while (avcodec_receive_frame(audio_ctx, af) >= 0) {

            int out_channels = out_ch_layout.nb_channels;
            int bps         = av_get_bytes_per_sample(out_format);

            int max_out_samples = av_rescale_rnd(
                swr_get_delay(swr, in_sample_rate) + af->nb_samples,
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
                (const uint8_t**)af->extended_data,
                af->nb_samples
                );

            if (audio_buffer_size < out_samples * 4 * 2) {
                audio_buffer_size = out_samples * 4 * 2;
                audio_buffer = (uint8_t*)realloc(audio_buffer, audio_buffer_size);
            }

            int samples = swr_convert(
                swr,
                &audio_buffer,
                out_samples,
                (const uint8_t**)af->extended_data,
                af->nb_samples
                );

            int bytes = samples * 2 * av_get_bytes_per_sample(out_format);

            if(!encode_flag){
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
        av_frame_free(&af);
    }
    av_packet_unref(pkt);
}

//CUDAで映像フレームを処理
void nvgpudecode::CUDA_RGBA_to_merge(){
    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    //本カーネル
    if (vd.size() == 2) {
        //NV12→RGBA→結合
        CUDA_IMG_Proc->nv12x2_to_rgba_merge(
            vd[0].Frame->data[0],vd[0].Frame->linesize[0], vd[0].Frame->data[1],vd[0].Frame->linesize[1],
            vd[1].Frame->data[0],vd[1].Frame->linesize[0], vd[1].Frame->data[1],vd[1].Frame->linesize[1],
            d_rgba, pitch_rgba,VideoInfo.width*2,VideoInfo.height,VideoInfo.width,VideoInfo.height,stream);
    }else if (vd.size() == 4) {
        //NV12→RGBA→結合
        CUDA_IMG_Proc->nv12x4_to_rgba_merge(
            vd[0].Frame->data[0],vd[0].Frame->linesize[0], vd[0].Frame->data[1],vd[0].Frame->linesize[1],
            vd[1].Frame->data[0],vd[1].Frame->linesize[0], vd[1].Frame->data[1],vd[1].Frame->linesize[1],
            vd[2].Frame->data[0],vd[2].Frame->linesize[0], vd[2].Frame->data[1],vd[2].Frame->linesize[1],
            vd[3].Frame->data[0],vd[3].Frame->linesize[0], vd[3].Frame->data[1],vd[3].Frame->linesize[1],
            d_rgba, pitch_rgba,VideoInfo.width*2,VideoInfo.height*2,VideoInfo.width,VideoInfo.height,stream);
    }else{
        // NV12 → RGBA
        if(VideoInfo.bitdepth == 8){
            CUDA_IMG_Proc->NV12_to_RGBA_8bit(
                d_rgba,
                pitch_rgba,
                vd[0].Frame->data[0],
                vd[0].Frame->linesize[0],
                vd[0].Frame->data[1],
                vd[0].Frame->linesize[1],
                VideoInfo.width,
                VideoInfo.height,
                stream
                );
        }else if(VideoInfo.bitdepth == 10){
            CUDA_IMG_Proc->NV12_to_RGBA_10bit(
                d_rgba,
                pitch_rgba,
                vd[0].Frame->data[0],
                vd[0].Frame->linesize[0],
                vd[0].Frame->data[1],
                vd[0].Frame->linesize[1],
                VideoInfo.width,
                VideoInfo.height,
                stream
                );
        }
    }

    //本カーネル同期
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    //ダミーカーネルで完全な同期
    CUDA_IMG_Proc->Dummy(stream);
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    //フレーム番号取得
    VideoInfo.current_frameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;
    slider_No = VideoInfo.current_frameNo;

    emit send_decode_image(d_rgba,  pitch_rgba, VideoInfo.current_frameNo);
}
