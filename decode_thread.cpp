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
        if (codec_ctx) {
            if (codec_ctx->codec && codec_ctx->internal)
                avcodec_flush_buffers(codec_ctx);
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
    };

    auto safe_free_format = [&]() {
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    };

    auto safe_free_frames = [&]() {
        if (hw_frame) {
            av_frame_free(&hw_frame);
            hw_frame = nullptr;
        }
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
    };

    auto safe_free_hwctx = [&]() {
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }
    };

    try {
        safe_free_codec();
        safe_free_format();
        safe_free_frames();
        safe_free_hwctx();
    } catch (...) {
        qWarning() << "Exception during FFmpeg cleanup (ignored)";
    }

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

    safe_cuda_free((void*&)d_rgba, "d_rgba");

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
    if (!video_play_flag && slider_No == VideoInfo.current_frameNo){
        if(decode_state==STATE_DECODE_READY){
            decode_state=STATE_DECODING;
            emit send_decode_image(nullptr,0,VideoInfo.current_frameNo);
            decode_state=STATE_WAIT_DECODE_FLAG;
        }
        return;
    }

    if(decode_state==STATE_DECODE_READY){
        decode_state=STATE_DECODING;
        get_decode_image();
        decode_state=STATE_WAIT_DECODE_FLAG;
    }

    //デコード修了指示が出た場合は全ての処理を完了してから修了を通知
    if(thread_stop_flag){
        emit finished();
    }
}

// FFmpeg 初期化
void decode_thread::initialized_ffmpeg() {
    packet = av_packet_alloc();
    hw_frame = av_frame_alloc();
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

    video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        QString error= "Could not find video stream";
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    //フレームレートを取得
    VideoInfo.fps = getFrameRate(fmt_ctx, video_stream_index);

    //デコーダ設定
    const char* codec_name = avcodec_get_name(fmt_ctx->streams[video_stream_index]->codecpar->codec_id);
    const char* decoder_name = selectDecoder(codec_name);

    if (!decoder_name) {
        QString error= "Unsupported codec:"+QString::fromStdString(codec_name);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    decoder = avcodec_find_decoder_by_name(decoder_name);
    if (!decoder) {
        QString error= "Decoder not found:" + QString::fromStdString(decoder_name);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "split_decode_mode", "1", 0);

    codec_ctx = avcodec_alloc_context3(decoder);
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_index]->codecpar);

    ret = avcodec_open2(codec_ctx, decoder, &opts);
    if (ret < 0) {
        QString error= "Codec open error (" + QString::fromStdString(decoder_name) + "):" + ffmpegErrStr(ret);
        qDebug()<<error;
        emit decode_error(error);
        return;
    }

    //1フレームのPTSを計算
    double time_base_d = av_q2d(fmt_ctx->streams[video_stream_index]->time_base);
    VideoInfo.pts_per_frame = 1.0 / (VideoInfo.fps * time_base_d);
    qDebug() << "1フレームのPTS数:" << VideoInfo.pts_per_frame;

    //最終フレームptsを取得
    get_last_frame_pts();

    //スライダー設定
    emit send_video_info();

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
    avcodec_flush_buffers(codec_ctx);

    int64_t duration_pts = fmt_ctx->streams[video_stream_index]->duration;
    if (duration_pts <= 0) {
        // durationが不明な場合はファイル末尾にシーク
        duration_pts = INT64_MAX;
    }

    // 後方シーク（できるだけ終端に近づく）
    if (av_seek_frame(fmt_ctx, video_stream_index, duration_pts, AVSEEK_FLAG_BACKWARD) < 0) {
        qDebug() << "seek failed";
        return;
    }

    avcodec_flush_buffers(codec_ctx);

    int64_t last_pts = AV_NOPTS_VALUE;
    bool frame_received = false;

    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);
        if (ret < 0) {
            // EOFに到達：デコーダに残りを流す
            avcodec_send_packet(codec_ctx, nullptr);
            while (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                last_pts = hw_frame->best_effort_timestamp;
                VideoInfo.width = hw_frame->width;
                VideoInfo.height = hw_frame->height;
                frame_received = true;
            }
            break;
        }

        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                    last_pts = hw_frame->best_effort_timestamp;
                    VideoInfo.width = hw_frame->width;
                    VideoInfo.height = hw_frame->height;
                    frame_received = true;
                }
            }
        }
        av_packet_unref(packet);
    }

    if (frame_received) {
        AVRational tb = fmt_ctx->streams[video_stream_index]->time_base;
        double seconds = last_pts * av_q2d(tb);
        qDebug() << "Last PTS:" << last_pts << " (" << seconds << "sec)";
        VideoInfo.max_framesNo = fmt_ctx->streams[video_stream_index]->duration/VideoInfo.pts_per_frame-1;
        VideoInfo.current_frameNo = VideoInfo.max_framesNo;

        //初回時にのみmalloc
        cudaMallocPitch(&d_rgba, &pitch_rgba,VideoInfo.width*4, VideoInfo.height);
    } else {
        qDebug() << "No frame found at end.";
    }
}

// フレーム取得
void decode_thread::get_decode_image() {
    // ----------- シーク処理 -----------
    if (slider_No != VideoInfo.current_frameNo || video_reverse_flag) {

        if (video_reverse_flag) {
            slider_No--;
            if (slider_No < 0)
                slider_No = VideoInfo.max_framesNo;
        }

        // ビデオ/オーディオの両方 flush
        avcodec_flush_buffers(codec_ctx);
        if (audio_ctx)
            avcodec_flush_buffers(audio_ctx);

        av_seek_frame(fmt_ctx, video_stream_index,
                      slider_No * VideoInfo.pts_per_frame,
                      AVSEEK_FLAG_BACKWARD);

        VideoInfo.current_frameNo = slider_No;
    }

    // ----------- パケット読み込みループ -----------
    while (true) {
        int ret = av_read_frame(fmt_ctx, packet);

        // ---------- EOF ----------
        if (ret < 0) {

            // 映像側フラッシュ
            avcodec_send_packet(codec_ctx, nullptr);

            if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                ffmpeg_to_CUDA();
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

            avcodec_flush_buffers(codec_ctx);
            if (audio_ctx)
                avcodec_flush_buffers(audio_ctx);

            av_seek_frame(fmt_ctx, video_stream_index,
                          seek_frame * VideoInfo.pts_per_frame,
                          AVSEEK_FLAG_ANY);
            continue;
        }

        // ----------- AUDIO PACKET -----------
        if (packet->stream_index == audio_stream_index) {

            if (avcodec_send_packet(audio_ctx, packet) >= 0) {
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
            av_packet_unref(packet);
            continue;  // → 映像パケットを探しに行く
        }

        // ----------- VIDEO PACKET -----------
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0) {
                av_packet_unref(packet);
                continue;
            }

            if (avcodec_receive_frame(codec_ctx, hw_frame) == 0) {
                ffmpeg_to_CUDA();
                av_packet_unref(packet);
                break;  // 映像が取れたら終了
            }
        }

        av_packet_unref(packet);
    }

    // 念のため
    av_packet_unref(packet);
}


//CUDAに渡して画像処理
void decode_thread::ffmpeg_to_CUDA(){
    // QElapsedTimer timer;
    // timer.start();

    //カーネルNV12→RGBA
    CUDA_IMG_Proc->NV12_to_RGBA(d_rgba, pitch_rgba,hw_frame->data[0], hw_frame->linesize[0], hw_frame->data[1], hw_frame->linesize[1], VideoInfo.height, VideoInfo.width);

    //フレーム番号更新
    AVRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
    VideoInfo.current_frameNo = hw_frame->best_effort_timestamp/VideoInfo.pts_per_frame;
    slider_No = VideoInfo.current_frameNo;

    emit send_decode_image(d_rgba,pitch_rgba,VideoInfo.current_frameNo);
    // double seconds = timer.nsecsElapsed() / 1e6; // ナノ秒 →  ミリ秒
    // qDebug()<<seconds;
}





