#include "src/videoprocess/nvgpudecode.h"

// ===============================
// mp4 NVIDIA向けGPUデコード
// ===============================

// ffmpeg初期化
bool nvgpudecode::initialized_ffmpeg()
{
    packet = av_packet_alloc();
    int ret;

    // ------------------------
    // CUDA デバイスコンテキスト作成
    // ------------------------
    QString gpuId = QString::number(g_cudaDeviceID);
    qDebug() << "GPU No:" << gpuId;

    ret = av_hwdevice_ctx_create(
        &hw_device_ctx,
        AV_HWDEVICE_TYPE_CUDA,
        gpuId.toUtf8().data(),
        nullptr,
        0
        );

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

    // ------------------------
    // デコーダ作成（GPU）
    // ------------------------
    for (int i = 0; i < video_stream_indices.size(); i++) {
        int stream_index = video_stream_indices[i];

        vd.emplace_back();
        auto& dec = vd.back();

        dec.stream_index = stream_index;
        dec.Frame = av_frame_alloc();
        if (!dec.Frame) {
            Error_String = "av_frame_alloc failed";
            return false;
        }

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

        dec.codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        dec.codec_ctx->pkt_timebase = st->time_base;

        ret = avcodec_parameters_to_context(dec.codec_ctx, st->codecpar);
        if (ret < 0) {
            Error_String = QString("avcodec_parameters_to_context failed: %1")
            .arg(ffmpegErrStr(ret));
            return false;
        }

        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "extra_hw_frames", "0", 0);

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

    // ------------------------
    // CUDA メモリ確保
    // ------------------------
    cudaError_t err = cudaMallocPitch(
        &Frame.d_decode_rgba,
        &Frame.decode_pitch,
        VideoInfo.width * VideoInfo.width_scale * 4,
        VideoInfo.height * VideoInfo.height_scale
        );
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }
    err = cudaMallocPitch(
        &Frame.d_encode_rgba,
        &Frame.encode_pitch,
        VideoInfo.width * VideoInfo.width_scale * 4,
        VideoInfo.height * VideoInfo.height_scale
        );
    if (err != cudaSuccess) {
        Error_String = QString("cudaMallocPitch failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

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

        audioSink = new QAudioSink(fmt);
        audioOutput = audioSink->start();

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
        qDebug()<<codec;
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc_cuvid";
        qDebug()<<codec;
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1_cuvid";
        qDebug()<<codec;
    }else{
        codec=nullptr;
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

//テストでコード、最終フレームpts取得
bool nvgpudecode::get_last_frame_pts() {
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

    av_packet_unref(packet);

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double time = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << time << "sec)";
        VideoInfo.max_framesNo = last_pts/VideoInfo.pts_per_frame;;
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

//映像シングルストリーム
void nvgpudecode::get_singledecode_image() {
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

            if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
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
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                    av_packet_unref(packet);
                    break;
                }
            }
        }
        av_packet_unref(packet);
    }

    CUDA_RGBA_to_merge();
    av_packet_unref(packet);
}

//映像マルチストリーム
void nvgpudecode::get_multidecode_image() {
    std::vector<bool> got_frame(vd.size(), false);
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

    while (got_count < vd.size()) {
        int ret = av_read_frame(fmt_ctx, packet);

        if (ret < 0) {
            for (int i = 0; i < vd.size(); i++) {
                avcodec_send_packet(vd[i].codec_ctx, nullptr);
                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                    got_frame[i] = true;
                    got_count++;
                }
            }

            if (Frame.FrameNo >= VideoInfo.end_range_framesNo) {
                emit decode_end();
                high_res_seek_frame(VideoInfo.start_range_framesNo,false);
                continue;
            }

            av_packet_unref(packet);
            break;
        }

        if (packet->stream_index == audio_stream_index) {
            get_decode_audio();
            av_packet_unref(packet);
            continue;
        }

        for (int i = 0; i < vd.size(); i++) {
            if (packet->stream_index == vd[i].stream_index && !got_frame[i]) {
                if (avcodec_send_packet(vd[i].codec_ctx, packet) == 0) {
                    if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                        got_frame[i] = true;
                        got_count++;
                    }
                }
            }
        }

        av_packet_unref(packet);
    }

    // CUDAの処理,NVDEC完了待ち
    cudaEventRecord(events, stream);
    cudaEventSynchronize(events);

    CUDA_RGBA_to_merge();
}

//オーディオ
void nvgpudecode::get_decode_audio()
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
            if (audio_mode && audioOutput) {
                if (audioSink->bytesFree() >= pcm.size()) {

                    float volume = g_audio_vol / 100.0f;

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
            Frame.d_decode_rgba,Frame.decode_pitch,VideoInfo.width*VideoInfo.width_scale,VideoInfo.height*VideoInfo.height_scale,VideoInfo.width,VideoInfo.height,stream);
    }else if (vd.size() == 4) {
        //NV12→RGBA→結合
        CUDA_IMG_Proc->nv12x4_to_rgba_merge(
            vd[0].Frame->data[0],vd[0].Frame->linesize[0], vd[0].Frame->data[1],vd[0].Frame->linesize[1],
            vd[1].Frame->data[0],vd[1].Frame->linesize[0], vd[1].Frame->data[1],vd[1].Frame->linesize[1],
            vd[2].Frame->data[0],vd[2].Frame->linesize[0], vd[2].Frame->data[1],vd[2].Frame->linesize[1],
            vd[3].Frame->data[0],vd[3].Frame->linesize[0], vd[3].Frame->data[1],vd[3].Frame->linesize[1],
            Frame.d_decode_rgba, Frame.decode_pitch,VideoInfo.width*VideoInfo.width_scale,VideoInfo.height*VideoInfo.height_scale,VideoInfo.width,VideoInfo.height,stream);
    }else if(vd.size() == 8){
        //NV12→RGBA→結合
        CUDA_IMG_Proc->nv12x8_to_rgba_merge(
            vd[0].Frame->data[0],vd[0].Frame->linesize[0], vd[0].Frame->data[1],vd[0].Frame->linesize[1],
            vd[1].Frame->data[0],vd[1].Frame->linesize[0], vd[1].Frame->data[1],vd[1].Frame->linesize[1],
            vd[2].Frame->data[0],vd[2].Frame->linesize[0], vd[2].Frame->data[1],vd[2].Frame->linesize[1],
            vd[3].Frame->data[0],vd[3].Frame->linesize[0], vd[3].Frame->data[1],vd[3].Frame->linesize[1],
            vd[4].Frame->data[0],vd[4].Frame->linesize[0], vd[4].Frame->data[1],vd[4].Frame->linesize[1],
            vd[5].Frame->data[0],vd[5].Frame->linesize[0], vd[5].Frame->data[1],vd[5].Frame->linesize[1],
            vd[6].Frame->data[0],vd[6].Frame->linesize[0], vd[6].Frame->data[1],vd[6].Frame->linesize[1],
            vd[7].Frame->data[0],vd[7].Frame->linesize[0], vd[7].Frame->data[1],vd[7].Frame->linesize[1],
            Frame.d_decode_rgba, Frame.decode_pitch,VideoInfo.width*VideoInfo.width_scale,VideoInfo.height*VideoInfo.height_scale,VideoInfo.width,VideoInfo.height,stream);
    }else{
        // NV12 → RGBA
        if(VideoInfo.bitdepth == 8){
            CUDA_IMG_Proc->NV12_to_RGBA_8bit(
                Frame.d_decode_rgba,
                Frame.decode_pitch,
                vd[0].Frame->data[0],
                vd[0].Frame->linesize[0],
                vd[0].Frame->data[1],
                vd[0].Frame->linesize[1],
                VideoInfo.width*VideoInfo.width_scale,
                VideoInfo.height*VideoInfo.height_scale,
                stream
                );
        }else if(VideoInfo.bitdepth == 10){
            CUDA_IMG_Proc->NV12_to_RGBA_10bit(
                Frame.d_decode_rgba,
                Frame.decode_pitch,
                vd[0].Frame->data[0],
                vd[0].Frame->linesize[0],
                vd[0].Frame->data[1],
                vd[0].Frame->linesize[1],
                VideoInfo.width*VideoInfo.width_scale,
                VideoInfo.height*VideoInfo.height_scale,
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
    if(seek_flag){
        Frame.FrameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;
        slider_No = Frame.FrameNo;
        back1FrameNo = Frame.FrameNo-1;
    }else if(back1frame_flag||high_res_slider_flag){
        Frame.FrameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;
        slider_No = Frame.FrameNo;
    }else{
        back1FrameNo = Frame.FrameNo;
        Frame.FrameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;
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
void nvgpudecode::high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag){
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

    if(vd.size()==0){
        return;
    }else if(vd.size()==1){
        high_res_seek_frame_single(targetFrameNo);
    }else{
        high_res_seek_frame_multi(targetFrameNo);
    }
}

//高精度シークシングルストリーム
void nvgpudecode::high_res_seek_frame_single(int targetFrameNo){
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
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
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
                    if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].Frame) == 0) {
                        av_packet_unref(packet);
                        break;
                    }
                }
            }
        }

        //ターゲットフレームの番号かどうか判定
        back1FrameNo = FrameNo;
        FrameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo <= FrameNo){
            CUDA_RGBA_to_merge();
            break;
        }
        av_packet_unref(packet);
    }
}

//高精度シークマルチストリーム
void nvgpudecode::high_res_seek_frame_multi(int targetFrameNo){
    // ----------- シーク処理 -----------
    // ビデオ/オーディオの両方 flush
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    for (auto& dec : vd) {
        avcodec_flush_buffers(dec.codec_ctx);
    }

    av_seek_frame(fmt_ctx, vd[0].stream_index,
                  targetFrameNo * VideoInfo.pts_per_frame,
                  AVSEEK_FLAG_BACKWARD);

    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }
    while(!thread_stop_flag){
        std::vector<bool> got_frame(vd.size(), false);
        int got_count = 0;

        while (got_count < vd.size()) {
            int ret = av_read_frame(fmt_ctx, packet);

            if (ret < 0) {
                for (int i = 0; i < vd.size(); i++) {
                    avcodec_send_packet(vd[i].codec_ctx, nullptr);
                    if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                        got_frame[i] = true;
                        got_count++;
                    }
                }

                av_packet_unref(packet);
                break;
            }

            for (int i = 0; i < vd.size(); i++) {
                if (packet->stream_index == vd[i].stream_index && !got_frame[i]) {
                    if (avcodec_send_packet(vd[i].codec_ctx, packet) == 0) {
                        if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].Frame) == 0) {
                            got_frame[i] = true;
                            got_count++;
                        }
                    }
                }
            }

            av_packet_unref(packet);
        }

        // CUDAの処理,NVDEC完了待ち
        cudaEventRecord(events, stream);
        cudaEventSynchronize(events);

        //フレーム番号記憶
        back1FrameNo = FrameNo;
        FrameNo = vd[0].Frame->best_effort_timestamp / VideoInfo.pts_per_frame;

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo<=FrameNo){
            CUDA_RGBA_to_merge();
            break;
        }
        av_packet_unref(packet);
    }
}
