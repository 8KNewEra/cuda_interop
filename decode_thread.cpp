#include "decode_thread.h"
#include "qfileinfo.h"
#include <QDebug>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QDebug>
#include <QTimer>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
#include <libavutil/error.h>
}

decode_thread::decode_thread(QString FilePath,bool audio_m,QObject *parent)
    : QObject(parent), video_play_flag(true), timer(new QTimer(this))  {
    QFileInfo fileInfo(FilePath);
    VideoInfo.Name = fileInfo.completeBaseName().toStdString()+".mp4";

    VideoInfo.Path = FilePath.toStdString();
    File_byteArray = FilePath.toUtf8();
    input_filename = File_byteArray.constData();

    if(CUDA_IMG_Proc==nullptr){
        CUDA_IMG_Proc=new CUDA_ImageProcess();
    }

    audio_mode=audio_m;
}

decode_thread::~decode_thread() {
    if (timer) {
        QMetaObject::invokeMethod(timer, "stop", Qt::QueuedConnection);
        QMetaObject::invokeMethod(timer, "deleteLater", Qt::QueuedConnection);
        timer = nullptr;
        delete timer;
    }

    // 1) デコードループ停止要求
    QThread::msleep(10);

    // 2) CUDA同期（非同期処理完了待ち）
    cudaError_t cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        qWarning() << "cudaDeviceSynchronize failed:"
                   << cudaGetErrorString(cerr);
    }

    // 3) FFmpeg関係の安全な解放
    auto safe_free_codec = [&]() {
        for (int i = 0; i < vd.size()-1; i++) {
            if (vd[i].codec_ctx) {
                if (vd[i].codec_ctx->codec && vd[i].codec_ctx->internal)
                    avcodec_flush_buffers(vd[i].codec_ctx);
                avcodec_free_context(&vd[i].codec_ctx);
                vd[i].codec_ctx = nullptr;
            }
        }
    };

    auto safe_free_format = [&]() {
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    };

    auto safe_free_frames = [&]() {
        for (int i = 0; i < vd.size()-1; i++) {
            if (vd[i].hw_frame) {
                av_frame_free(&vd[i].hw_frame);
                vd[i].hw_frame = nullptr;
            }
            if (vd[i].packet) {
                av_packet_free(&vd[i].packet);
                vd[i].packet = nullptr;
            }
        }
    };

    auto safe_free_hwctx = [&]() {
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }
    };

    // 4) CUDA メモリの安全な解放
    auto safe_cuda_free = [&](void*& ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess)
                qWarning() << name << "cudaFree failed:"
                           << cudaGetErrorString(err);
            ptr = nullptr;
        }
    };

    for (int i = 0; i <vd.size(); i++) {
        if(stream[i]){
            cudaStreamDestroy(stream[i]);
            stream[i]=nullptr;
        }


        for (int b = 0; b < BUF; b++) {
            if(d_rgba[i][b]){
                safe_cuda_free((void*&)d_rgba[i][b], "d_rgba");
            }

            if (events[i][b]) {
                cudaEventDestroy(events[i][b]);
                events[i][b] = nullptr;
            }
        }
    }

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
    } catch (...) {
        qWarning() << "Exception during FFmpeg cleanup (ignored)";
    }

    if(d_merged){
        safe_cuda_free((void*&)d_merged, "d_merged");
    }

    delete CUDA_IMG_Proc;
    CUDA_IMG_Proc=nullptr;

    qDebug() << "decode_thread: resources released cleanly";
}

QString decode_thread::ffmpegErrStr(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(errnum, buf, sizeof(buf));
    return QString::fromUtf8(buf);
}

void decode_thread::receve_decode_flag(){
    QMutexLocker locker(&mutex);
    if(decode_state==STATE_WAIT_DECODE_FLAG){
        decode_state = STATE_DECODE_READY;
    }
}

void decode_thread::set_decode_speed(int speed){
    interval_ms = static_cast<int>(1000.0 / speed);
    elapsedTimer.start();
    timer->start(interval_ms);
}

void decode_thread::startProcessing() {
    initialized_ffmpeg();
    qDebug() << "Live Thread: Start Processing";
}

void decode_thread::stopProcessing() {
    thread_stop_flag = true;

    qDebug() << "decode_thread: stopProcessing called";
}

void decode_thread::sliderPlayback(int value){
    pausePlayback();
    slider_No=value;
    video_play_flag = false;
    video_reverse_flag = false;
}

void decode_thread::resumePlayback() {
    receve_decode_flag();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = false;
}

void decode_thread::pausePlayback() {
    QMutexLocker locker(&mutex);
    video_play_flag = false;
}

void decode_thread::reversePlayback(){
    receve_decode_flag();
    QMutexLocker locker(&mutex);
    video_play_flag = true;
    video_reverse_flag = true;
}

void decode_thread::processFrame() {
    QMutexLocker locker(&mutex);

    //停止ボタン押下でシークしていない場合は停止
    if (!video_play_flag && slider_No == current_FrameNo){
        if(decode_state==STATE_DECODE_READY){
            decode_state=STATE_DECODING;
            emit send_decode_image(nullptr,0,current_FrameNo);
            decode_state=STATE_WAIT_DECODE_FLAG;
        }
        return;
    }

    if(decode_state==STATE_DECODE_READY){
        decode_state=STATE_DECODING;

        //複数ストリームとシングルストリームで別ルートを用意
        if(vd.size()==1){
            get_singlestream_gpudecode_image(0);
        }else{
            get_multistream_decode_image();
        }

        decode_state=STATE_WAIT_DECODE_FLAG;
    }

    //デコード修了指示が出た場合は全ての処理を完了してから修了を通知
    if(thread_stop_flag){
        emit finished();
    }
}

// FFmpeg 初期化
void decode_thread::initialized_ffmpeg() {
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

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        QString error="Failed to retrieve input stream information:"+ ffmpegErrStr(ret);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

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


    for (int i = 0; i < video_stream_indices.size(); i++) {
        int stream_index = video_stream_indices[i];

        vd.emplace_back();
        auto& dec = vd.back();

        dec.stream_index = stream_index;
        dec.packet   = av_packet_alloc();
        dec.hw_frame = av_frame_alloc();

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

    //フレームレートを取得
    VideoInfo.fps = getFrameRate(fmt_ctx, vd[0].stream_index);

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[vd[0].stream_index]->time_base);
    VideoInfo.pts_per_frame = 1.0 / (VideoInfo.fps * time_base_d);
    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    //最終フレームptsを取得
    get_last_frame_pts();

    //再生
    video_play_flag = true;
    video_reverse_flag = false;
    slider_No=0;

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
    emit send_video_info();

    //CUDAストリーム生成
    int N = vd.size();

    d_rgba.resize(N);
    pitch_rgba.resize(N);
    events.resize(N);
    stream.resize(N);

    write_idx.resize(N, 0);
    ready_idx.resize(N, -1);
    frame_no.resize(N, -1);
    pkt.resize(N, nullptr);
    rgba_finish.resize(N, false);

    for (int i = 0; i < N; i++) {
        cudaStreamCreateWithFlags(
            &stream[i],
            cudaStreamNonBlocking
            );

        d_rgba[i].resize(BUF);
        pitch_rgba[i].resize(BUF);
        events[i].resize(BUF);

        for (int b = 0; b < BUF; b++) {
            cudaError_t err = cudaMallocPitch(
                &d_rgba[i][b],
                &pitch_rgba[i][b],
                VideoInfo.width * 4,
                VideoInfo.height
                );

            if (err != cudaSuccess) {
                qDebug() << "cudaMallocPitch failed"
                         << i << b
                         << cudaGetErrorString(err);
            }

            // ✅ event 作成
            cudaEventCreateWithFlags(
                &events[i][b],
                cudaEventDisableTiming
                );
        }
    }


    qDebug()<<"AAAAA"<<VideoInfo.width<<":"<<VideoInfo.height;

    //デコードモード
    if(vd.size()==1){
        VideoInfo.decode_mode = "Decode Mode:Non split(tile:1)\n";
        VideoInfo.width_scale=1;
        VideoInfo.height_scale=1;
    }else if(vd.size()==2){
        VideoInfo.decode_mode = "Decode Mode:split_x2(tile:2×1)\n";
        VideoInfo.width_scale=2;
        VideoInfo.height_scale=1;
        cudaMallocPitch(&d_merged, &pitch_merged,VideoInfo.width*VideoInfo.width_scale*4, VideoInfo.height*VideoInfo.height_scale);
    }else if(vd.size()==4){
        VideoInfo.decode_mode = "Decode Mode:split_x4(tile:2×2)\n";
        VideoInfo.width_scale=2;
        VideoInfo.height_scale=2;
        cudaMallocPitch(&d_merged, &pitch_merged,VideoInfo.width*VideoInfo.width_scale*4, VideoInfo.height*VideoInfo.height_scale);
    }

    interval_ms = static_cast<double>(1000.0 / 33);
    connect(timer, &QTimer::timeout, this, &decode_thread::processFrame);
    elapsedTimer.start();
    timer->start(interval_ms);
}

// デコーダ設定
const char*decode_thread::selectDecoder(const char* codec_name) {
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
double decode_thread::getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index) {
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

void decode_thread::get_last_frame_pts() {
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
        int ret = av_read_frame(fmt_ctx, vd[0].packet);
        if (ret < 0) {
            // EOFに到達：デコーダに残りを流す
            avcodec_send_packet(vd[0].codec_ctx, nullptr);
            while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frame) == 0) {
                last_pts = vd[0].hw_frame->best_effort_timestamp;
                VideoInfo.width = vd[0].hw_frame->width;
                VideoInfo.height = vd[0].hw_frame->height;
                frame_received = true;
            }
            break;
        }

        if (vd[0].packet->stream_index == vd[0].stream_index) {
            if (avcodec_send_packet(vd[0].codec_ctx, vd[0].packet) == 0) {
                while (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frame) == 0) {
                    last_pts = vd[0].hw_frame->best_effort_timestamp;
                    VideoInfo.width = vd[0].hw_frame->width;
                    VideoInfo.height = vd[0].hw_frame->height;
                    frame_received = true;
                }
            }
        }
        av_packet_unref(vd[0].packet);
    }

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[vd[0].stream_index]->time_base;
        double seconds = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << seconds << "sec)";
        VideoInfo.max_framesNo = (fmt_ctx->streams[vd[0].stream_index]->duration/VideoInfo.pts_per_frame-1);
        current_FrameNo = VideoInfo.max_framesNo;
    } else {
        qDebug() << "No frame found at end.";
    }
}

// 複数ストリームフレーム取得
void decode_thread::get_multistream_decode_image() {
    std::vector<bool> got_frame(vd.size(), false);
    int got_count = 0;

    //----------- シーク処理 -----------
    if (slider_No != current_FrameNo || video_reverse_flag) {
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

        current_FrameNo = slider_No;

        a+=1;
    }

    while (got_count < vd.size()) {
        pkt[got_count]=av_packet_alloc();
        int ret = av_read_frame(fmt_ctx, pkt[got_count]);
        // ---------- EOF ----------
        if (ret < 0) {
            for (int i = 0; i < vd.size(); i++) {
                // 映像側フラッシュ
                avcodec_send_packet(vd[i].codec_ctx, nullptr);

                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frame) == 0) {
                    got_frame[i] = true;
                    got_count++;
                }
            }

            // 終端→先頭に戻ってループ
            uint64_t seek_frame;
            if (VideoInfo.current_frameNo >= VideoInfo.max_framesNo) {
                emit decode_end();
                seek_frame = 0;

                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                //複数ストリームは0でシーク
                for (auto& dec : vd) { avcodec_flush_buffers(dec.codec_ctx); }
                av_seek_frame(fmt_ctx, 0,
                              seek_frame * VideoInfo.pts_per_frame,
                              AVSEEK_FLAG_ANY);
                continue;
            }
        }

        // ----------- AUDIO PACKET -----------
        if (pkt[got_count]->stream_index == audio_stream_index) {
            get_decode_audio(pkt[got_count]);
            continue;  // → 映像パケットを探しに行く
        }

        // ----------- VIDEO PACKET -----------
        for (int i = 0; i < vd.size(); i++) {
            if (pkt[got_count]->stream_index == vd[i].stream_index&& !got_frame[i]) {
                if (avcodec_send_packet(vd[i].codec_ctx, pkt[got_count]) < 0) {
                    continue;
                }

                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frame) == 0) {
                    qDebug()<<i;
                    got_frame[i] = true;
                    got_count++;
                    break;  // 映像が取れたら終了
                }
            }
        }
        qDebug()<<"aaaaaaaaaaaaaa";

    }
    // 念のため
    CUDA_RGBA_to_merge();

    for(int i=0;i<vd.size();i++){
        av_packet_unref(pkt[i]);
    }

}

//シングルストリーム
void decode_thread::get_singlestream_gpudecode_image(int i){
    AVPacket* pkt = av_packet_alloc();

    // ----------- シーク処理 -----------
    if (slider_No != VideoInfo.current_frameNo || video_reverse_flag) {

        if (video_reverse_flag) {
            slider_No--;
            if (slider_No < 0)
                slider_No = VideoInfo.max_framesNo;
        }

        // ビデオ/オーディオの両方 flush
        avcodec_flush_buffers(vd[i].codec_ctx);
        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);

        av_seek_frame(fmt_ctx, vd[i].stream_index,
                      slider_No * VideoInfo.pts_per_frame,
                      AVSEEK_FLAG_BACKWARD);

        VideoInfo.current_frameNo = slider_No;
    }

    // ----------- パケット読み込みループ -----------
    while (true) {
        int ret = av_read_frame(fmt_ctx, pkt);

        // ---------- EOF ----------
        if (ret < 0) {

            // 映像側フラッシュ
            avcodec_send_packet(vd[i].codec_ctx, nullptr);

            if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frame) == 0) {
                ffmpeg_to_CUDA(i);
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

            avcodec_flush_buffers(vd[i].codec_ctx);
            if (audio_ctx)
                avcodec_flush_buffers(audio_ctx);

            av_seek_frame(fmt_ctx, vd[i].stream_index,
                          seek_frame * VideoInfo.pts_per_frame,
                          AVSEEK_FLAG_ANY);
            continue;
        }

        // ----------- AUDIO PACKET -----------
        if (pkt->stream_index == audio_stream_index) {
            get_decode_audio(pkt);
            continue;  // → 映像パケットを探しに行く
        }

        // ----------- VIDEO PACKET -----------
        if (pkt->stream_index == vd[i].stream_index) {
            if (avcodec_send_packet(vd[i].codec_ctx, pkt) < 0) {
                av_packet_unref(pkt);
                continue;
            }

            if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frame) == 0) {
                ffmpeg_to_CUDA(i);
                av_packet_unref(pkt);
                break;  // 映像が取れたら終了
            }
        }

        av_packet_unref(pkt);
    }
    // 念のため
    av_packet_unref(pkt);
}
//オーディオ
void decode_thread::get_decode_audio(AVPacket* pkt){
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

void decode_thread::CUDA_RGBA_to_merge(){
    if (vd.size() == 2) {

    }
    else if (vd.size() == 4) {
        CUDA_IMG_Proc->nv12x4_to_rgba_merge(
            vd[0].hw_frame->data[0],vd[0].hw_frame->linesize[0], vd[0].hw_frame->data[1],vd[0].hw_frame->linesize[1],
            vd[1].hw_frame->data[0],vd[1].hw_frame->linesize[0], vd[1].hw_frame->data[1],vd[1].hw_frame->linesize[1],
            vd[2].hw_frame->data[0],vd[2].hw_frame->linesize[0], vd[2].hw_frame->data[1],vd[2].hw_frame->linesize[1],
            vd[3].hw_frame->data[0],vd[3].hw_frame->linesize[0], vd[3].hw_frame->data[1],vd[3].hw_frame->linesize[1],
            d_merged, pitch_merged,VideoInfo.width*2,VideoInfo.height*2,VideoInfo.width,VideoInfo.height);

        //qDebug()<<vd[0].hw_frame->linesize[0]<<":"<<vd[0].hw_frame->linesize[1];

        cudaEventRecord(events[0][0], nullptr);
        cudaEventSynchronize(events[0][0]);

        current_FrameNo = vd[0].hw_frame->best_effort_timestamp / VideoInfo.pts_per_frame;
        VideoInfo.current_frameNo = current_FrameNo;
        slider_No = current_FrameNo;

        emit send_decode_image(d_merged, pitch_merged, current_FrameNo);
    }else{
        emit send_decode_image(d_rgba[0][ready_idx[0]],  pitch_rgba[0][ready_idx[0]], current_FrameNo);
    }
}






//CUDAに渡して画像処理
void decode_thread::ffmpeg_to_CUDA(int i)
{
    int64_t pts = vd[i].hw_frame->best_effort_timestamp;

    int b = write_idx[i];   // 今回使うバッファ

    // NV12 → RGBA
    CUDA_IMG_Proc->NV12_to_RGBA(
        d_rgba[i][b],
        pitch_rgba[i][b],
        vd[i].hw_frame->data[0],
        vd[i].hw_frame->linesize[0],
        vd[i].hw_frame->data[1],
        vd[i].hw_frame->linesize[1],
        VideoInfo.height,
        VideoInfo.width,
        stream[i]
        );

    cudaEventRecord(events[i][b], stream[i]);

    {
        QMutexLocker lock(&merge_mutex);
        frame_no[i]    = pts;
        ready_idx[i]   = b;
        rgba_finish[i] = true;
    }

    write_idx[i] = (b + 1) % BUF;

    if(i==0){
        qDebug()<<write_idx[0];
    }

    if (all_same_pts()) {
        for (int s = 0; s < vd.size(); s++) {
            cudaEventSynchronize(events[s][ready_idx[s]]);
        }

        current_FrameNo = frame_no[0] / VideoInfo.pts_per_frame;
        VideoInfo.current_frameNo = current_FrameNo;
        slider_No = current_FrameNo;

        // qDebug()<<vd[0].hw_frame->best_effort_timestamp<<":"<<vd[1].hw_frame->best_effort_timestamp<<":"<<vd[2].hw_frame->best_effort_timestamp<<":"<<vd[3].hw_frame->best_effort_timestamp;

        CUDA_merge();
    }
}


void decode_thread::CUDA_merge()
{
    if (vd.size() == 2) {
        CUDA_IMG_Proc->image_combine_x2(
            d_merged, pitch_merged,
            d_rgba[0][ready_idx[0]], pitch_rgba[0][ready_idx[0]],
            d_rgba[1][ready_idx[1]], pitch_rgba[1][ready_idx[1]],
            VideoInfo.width,
            VideoInfo.height
            );

        emit send_decode_image(d_merged, pitch_merged, current_FrameNo);
    }
    else if (vd.size() == 4) {
        CUDA_IMG_Proc->image_combine_x4(
            d_merged, pitch_merged,
            d_rgba[0][ready_idx[0]], pitch_rgba[0][ready_idx[0]],
            d_rgba[1][ready_idx[1]], pitch_rgba[1][ready_idx[1]],
            d_rgba[2][ready_idx[2]], pitch_rgba[2][ready_idx[2]],
            d_rgba[3][ready_idx[3]], pitch_rgba[3][ready_idx[3]],
            VideoInfo.width,
            VideoInfo.height,
            1
            );

        emit send_decode_image(d_merged, pitch_merged, current_FrameNo);
    }else{
        emit send_decode_image(d_rgba[0][ready_idx[0]],  pitch_rgba[0][ready_idx[0]], current_FrameNo);
    }

    for (int i = 0; i < vd.size(); i++) {
        rgba_finish[i] = false;
        frame_no[i]    = -1;
        ready_idx[i]   = -1;
    }
}



bool decode_thread::all_same_pts()
{
    if (!rgba_finish[0] || ready_idx[0] < 0)
        return false;

    int64_t pts0 = frame_no[0];

    for (int i = 1; i < vd.size(); i++) {
        if (!rgba_finish[i])      return false;
        if (ready_idx[i] < 0)     return false;
        if (frame_no[i] != pts0)  return false;
    }
    return true;
}








