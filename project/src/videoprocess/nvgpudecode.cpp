#include "src/videoprocess/nvgpudecode.h"

#include <algorithm>
#include <cstdlib>

// ===============================
// mp4 NVIDIA向けGPUデコード
// ===============================

// ffmpeg初期化
bool nvgpudecode::initialized_ffmpeg()
{
    //最初にGPU設定
    cudaSetDevice(g_openglDeviceID);

    packet = av_packet_alloc();
    int ret;

    // ------------------------
    // CUDA デバイスコンテキスト作成
    // ------------------------
    QString gpuId = QString::number(g_openglDeviceID);
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

        //リングバッファフレーム作成
        // ★ ゼロコピー化後は、このリングが「デコード結果の保護」そのものになる。
        //   GUIスレッドが hw_frames[j] を直接読むので、一周する前に
        //   上書きされないだけの面数が必要（in-flight は 3〜4 程度）。
        dec.hw_frames.resize(ringSize);
        for (int j = 0; j < ringSize; j++) {
            dec.hw_frames[j] = av_frame_alloc();
            if (!dec.hw_frames[j]) {
                Error_String = "av_frame_alloc failed";
                return false;
            }
        }

        dec.stream_index = stream_index;
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
        // ★ extra_hw_frames は「指定しない」。FFmpeg 既定の -1（自動）に任せる。
        //
        //   *_cuvid は hwaccel ではなく独立したデコーダで、出力サーフェスは
        //   自前のプール（surfaces オプション、既定 25 枚）から確保する。
        //   ここに extra_hw_frames を 1 以上で与えると
        //   ulNumDecodeSurfaces との整合が崩れ、デコード自体は通るのに
        //   シーク（avcodec_flush_buffers + av_seek_frame）後に
        //   "cuvid decode callback error" で落ちる。
        //   0 を明示するのも「追加ゼロ」という明示指示になるので外す。
        //
        //   アプリが hw_frames を ringSize(=6) 枚保持しても、
        //   既定 25 枚のプールに対しては十分余裕がある。
        //   どうしても枚数を明示したい場合は extra_hw_frames ではなく
        //   cuvid の surfaces を使うこと（extra_hw_frames とは併用しない）:
        //     av_dict_set_int(&opts, "surfaces", ringSize + 20, 0);

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
    // ★ ゼロコピー化により、中間の RGBA バッファは入口・出口とも不要。
    //   入口: NV12→RGBA結合はGUIスレッドがOpenGLテクスチャへ直接書き込む
    //   出口: NV12変換カーネルが FBO を texture object 経由で直読みする
    //   16K RGBA 1面 = 約531MB。入口4面 + 出口1面 = 約2.6GB の削減。

    // CUDA Stream
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        Error_String = QString("cudaStream failed: %1")
        .arg(QString::fromUtf8(cudaGetErrorString(err)));
        return false;
    }

    // CUDA Event
    err = cudaEventCreateWithFlags(&events, cudaEventDisableTiming);
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


// ==============================================================
// ★ パケット 1 本をストリーム i のデコーダへ渡し、出たフレームを受け取る
//
//   ★★ EAGAIN でパケットを捨てないこと。★★
//
//   cuvid は出力キューが埋まると avcodec_send_packet() が EAGAIN を返す。
//   以前は戻り値が 0 のときしか処理せず、そのまま av_packet_unref() で
//   捨てていたため、そのストリームだけフレームが 1 枚永久に欠けていた。
//   負荷が高いストリームほど EAGAIN を踏みやすいので、
//   「たまにどこかのタイルだけ取り残される」症状の原因になる。
//
//   EAGAIN が来たら 1 枚受け取ってキューを空け、必ず送り直す。
// ==============================================================
// ★ デストラクタ：保留パケットを解放する
// ==============================================================
nvgpudecode::~nvgpudecode()
{
    clear_pending();
}


// ==============================================================
// ★ 保留パケットの破棄
//   flush / シーク / 終了時に呼ぶ。残したまま flush すると、
//   シーク前のパケットをシーク後のデコーダへ入れてしまう。
// ==============================================================
void nvgpudecode::clear_pending()
{
    for (auto& q : pending_) {
        while (!q.empty()) {
            AVPacket* p = q.front();
            q.pop_front();
            av_packet_free(&p);
        }
    }
}


// ==============================================================
// ★ ファイルからパケットを 1 本読んで pending_ へ積む
//
//   ★★ ここでデコーダへは入れない。★★
//     デコーダに溜めると *_cuvid の出力サーフェス（自前プール、
//     既定 25 枚）を食い潰してデコードが止まる。
//     圧縮パケットをアプリ側で持つぶんには桁違いに軽い。
//
//   音声パケットはその場で処理して積まない。
//   戻り値 false = EOF
// ==============================================================
bool nvgpudecode::read_into_pending()
{
    if (pending_.size() != vd.size()) pending_.resize(vd.size());

    int ret = av_read_frame(fmt_ctx, packet);
    if (ret < 0) {
        av_packet_unref(packet);
        return false;                       // EOF
    }

    if (packet->stream_index == audio_stream_index) {
        get_decode_audio();
        av_packet_unref(packet);
        return true;
    }

    for (int i = 0; i < (int)vd.size(); i++) {
        if (packet->stream_index != vd[i].stream_index) continue;

        if ((int)pending_[i].size() >= kMaxPending) {
            // ここまで溜まるのは揃えられていない証拠。捨てずに警告だけ出し、
            // 一番古いものから落とす（落とさないとメモリが伸び続ける）。
            warn_once_in_a_while(
                QString("[nvgpudecode] tile%1 の保留パケットが上限 %2 に達しました")
                    .arg(i).arg(kMaxPending));
            AVPacket* old = pending_[i].front();
            pending_[i].pop_front();
            av_packet_free(&old);
        }

        AVPacket* clone = av_packet_clone(packet);
        if (clone) pending_[i].push_back(clone);
        break;
    }

    av_packet_unref(packet);
    return true;
}


// ==============================================================
// ★ pending_[i] から 1 本取り出してデコーダへ入れ、出た絵を受け取る
//
//   ★ 送ったら必ず受け取ること（入出力 1:1）。
//     受け取らずに溜めると cuvid の出力プールが枯渇する。
//   ★ EAGAIN でパケットを捨てないこと。捨てるとそのタイルだけ
//     フレームが欠けて、以降ずっとタイル間がずれる。
//     （これが元々のズレの原因だった）
//
//   戻り値 true = このストリームからフレームが 1 枚出た
// ==============================================================
bool nvgpudecode::pump_stream(int i, std::vector<bool>& got,
                              std::vector<int64_t>& pts, int& got_count)
{
    if (i < 0 || i >= (int)pending_.size() || pending_[i].empty()) return false;

    AVPacket* p = pending_[i].front();

    int r = avcodec_send_packet(vd[i].codec_ctx, p);

    if (r == AVERROR(EAGAIN)) {
        // 出力キューが満杯。1 枚受け取って空けてから送り直す。
        // ★ ここで pending_ から取り除かないこと（＝捨てない）
        if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frames[ringNo]) == 0) {
            pts[i] = vd[i].hw_frames[ringNo]->best_effort_timestamp;
            if (!got[i]) { got[i] = true; got_count++; }
            return true;
        }
        return false;
    }

    // 送れたので保留から外す
    pending_[i].pop_front();
    av_packet_free(&p);

    if (r < 0) return false;

    if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frames[ringNo]) == 0) {
        pts[i] = vd[i].hw_frames[ringNo]->best_effort_timestamp;
        if (!got[i]) { got[i] = true; got_count++; }
        return true;
    }
    return false;                           // まだ出力遅延の途中
}


// ==============================================================
// ★ 警告の連打を防ぐ
//   同じ原因で毎フレーム出すと、肝心のログが流れて読めなくなる。
// ==============================================================
void nvgpudecode::warn_once_in_a_while(const QString& msg)
{
    const int d = Frame.FrameNo - last_warn_frame_;
    if (d > -60 && d < 60) return;
    last_warn_frame_ = Frame.FrameNo;
    qWarning().noquote() << msg;
}


// ==============================================================
// ★ タイル間の pts を揃える（max 基準・順再生のまま）
//
//   一番進んでいるタイルを基準に、「遅れているタイルだけ」を前へ進める。
//   シークはしない。
//
//   進め方は 2 段階:
//     ① デコーダ内部に溜まっている出力をまず引き出す
//        （出力遅延の差ぶんは、これだけで追いつく）
//     ② それでも足りなければ、そのストリームの pending_ から
//        パケットを入れて進める
//
//   ★ 他のタイルには触らない。全員へ一斉に送ると全員が進むので
//     差が縮まらない。読んだパケットは pending_ に積むだけで、
//     デコーダへは入れない（プールを食わない）。
//
//   ★ 揃っている間のコストは int64 の比較 N 回だけで、そのまま return する。
// ==============================================================
bool nvgpudecode::align_tiles_pts(std::vector<int64_t>& pts)
{
    const int N = (int)vd.size();
    if (N <= 1) return true;
    if ((int)pending_.size() != N) pending_.resize(N);

    // 許容誤差は半フレーム
    const int64_t tol = (VideoInfo.pts_per_frame > 1.0)
                            ? (int64_t)(VideoInfo.pts_per_frame / 2.0)
                            : 0;

    std::vector<bool> dummy_got(N, true);   // ここでは got を進めない
    int dummy_count = N;

    for (int round = 0; round < kAlignMaxRounds; round++) {

        // ---- 基準 = 一番進んでいるタイル（max）----
        int64_t target = AV_NOPTS_VALUE;
        for (int i = 0; i < N; i++) {
            if (pts[i] == AV_NOPTS_VALUE) continue;
            if (target == AV_NOPTS_VALUE || pts[i] > target) target = pts[i];
        }
        if (target == AV_NOPTS_VALUE) return false;   // 1 枚も取れていない

        bool aligned   = true;
        bool advanced  = false;
        bool need_read = false;

        for (int i = 0; i < N; i++) {
            if (pts[i] != AV_NOPTS_VALUE && pts[i] + tol >= target) continue;

            // ① デコーダ内部の出力遅延ぶんを引き出す
            while (pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) {
                if (avcodec_receive_frame(vd[i].codec_ctx,
                                          vd[i].hw_frames[ringNo]) != 0) break;
                pts[i] = vd[i].hw_frames[ringNo]->best_effort_timestamp;
                advanced = true;
            }

            // ② 足りなければ、このストリームにだけパケットを入れて進める
            int spin = 0;
            while ((pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) &&
                   !pending_[i].empty() && spin < kLoopGuard) {
                if (pump_stream(i, dummy_got, pts, dummy_count)) advanced = true;
                spin++;
            }

            if (pts[i] == AV_NOPTS_VALUE || pts[i] + tol < target) {
                aligned = false;
                if (pending_[i].empty()) need_read = true;
            }
        }

        if (aligned) return true;

        // 追いつかせる材料が無いなら読み足す（pending_ に積むだけ）
        if (need_read) {
            if (!read_into_pending()) return false;   // EOF
            continue;
        }

        if (!advanced) return false;        // これ以上進めない
    }

    return false;
}


// ==============================================================
// ★ タイル間 pts 不一致の検出
//
//   align_tiles_pts() で揃えているので通常は出ない。
//   出た場合は「同じ表示フレームに別の絵のタイルが混ざっている」ので、
//   絵が割れる。再発を取り逃がさないための保険。
// ==============================================================
void nvgpudecode::check_tile_pts_mismatch()
{
    const int N = (int)vd.size();
    if (N <= 1) return;

    const int64_t tol = (VideoInfo.pts_per_frame > 1.0)
                            ? (int64_t)(VideoInfo.pts_per_frame / 2.0)
                            : 0;

    const int64_t base = vd[0].hw_frames[ringNo]->best_effort_timestamp;

    for (int i = 1; i < N; i++) {
        const int64_t p = vd[i].hw_frames[ringNo]->best_effort_timestamp;
        const int64_t d = (p > base) ? (p - base) : (base - p);
        if (d > tol) {
            warn_once_in_a_while(
                QString("[nvgpudecode] TILE PTS MISMATCH tile%1 pts=%2 vs tile0=%3 (%4 frames)")
                    .arg(i).arg(p).arg(base)
                    .arg((double)(p - base) / VideoInfo.pts_per_frame, 0, 'f', 1));
            break;
        }
    }
}


//映像シングルストリーム
void nvgpudecode::get_singledecode_image() {
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
                // ★ high_res_seek_frame の中で既に1フレーム emit しているので、
                //   ここでループを抜けて二重 emit を避ける
                av_packet_unref(packet);
                return;
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
            // ★ EAGAIN でパケットを捨てない（マルチと同じ理由）。
            //   単一ストリームではタイルズレにはならないが、
            //   そのぶんフレームが欠けてカクつきの原因になる。
            int r = avcodec_send_packet(vd[0].codec_ctx, packet);
            if (r == AVERROR(EAGAIN)) {
                if (avcodec_receive_frame(vd[0].codec_ctx,
                                          vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
                }
                r = avcodec_send_packet(vd[0].codec_ctx, packet);   // ★ 再送
            }
            if (r == 0) {
                if (avcodec_receive_frame(vd[0].codec_ctx,
                                          vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
                }
            }
        }
        av_packet_unref(packet);
    }

    publish_decode_frame();
    av_packet_unref(packet);
}

//映像マルチストリーム
void nvgpudecode::get_multidecode_image() {
    const int N = (int)vd.size();
    if ((int)pending_.size() != N) pending_.resize(N);

    std::vector<bool>    got_frame(N, false);
    std::vector<int64_t> pts(N, AV_NOPTS_VALUE);   // ★ 各タイルが掴んだ絵の pts
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
        clear_pending();          // ★ flush したら保留パケットも捨てる

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

    // -------------------------------------------------------------
    // ★ 各ストリームから 1 枚ずつ取り出す。
    //
    //   パケットは pending_ に積んでから、まだフレームを取れていない
    //   ストリームにだけ投入する。全員へ一斉に送らないのが要点で、
    //   これによって「遅れているタイルだけを進める」ことができる。
    //   投入したら必ず受け取る（cuvid のプールを食わないため）。
    // -------------------------------------------------------------
    bool eof = false;
    // ★ 空回りガード。pump_stream が EAGAIN で送れず受け取れもしない、
    //   あるいは映像パケットがいつまでも来ない場合に無限ループしないよう、
    //   読み足し回数で上限を切る。
    int read_guard = 0;
    while (got_count < N && read_guard < kLoopGuard) {

        // ① 保留分から、まだ取れていないストリームを進める
        for (int i = 0; i < N; i++) {
            int spin = 0;
            while (!got_frame[i] && !pending_[i].empty() && spin < kLoopGuard) {
                pump_stream(i, got_frame, pts, got_count);
                spin++;
            }
        }
        if (got_count >= N) break;

        // ② 足りなければ読み足す（pending_ に積むだけ）
        if (!read_into_pending()) { eof = true; break; }
        read_guard++;
    }

    if (read_guard >= kLoopGuard) {
        warn_once_in_a_while(
            QString("[nvgpudecode] フレームを揃えられずに打ち切りました FrameNo=%1 (取得 %2/%3)")
                .arg(Frame.FrameNo).arg(got_count).arg(N));
    }

    // ----------- EOF 処理 -----------
    if (eof) {
        for (int i = 0; i < N; i++) {
            avcodec_send_packet(vd[i].codec_ctx, nullptr);
            if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frames[ringNo]) == 0) {
                pts[i] = vd[i].hw_frames[ringNo]->best_effort_timestamp;
                if (!got_frame[i]) { got_frame[i] = true; got_count++; }
            }
        }

        if (Frame.FrameNo >= VideoInfo.end_range_framesNo) {
            emit decode_end();
            high_res_seek_frame(VideoInfo.start_range_framesNo,false);
            // ★ high_res_seek_frame の中で既に1フレーム emit しているので抜ける
            return;
        }
    }

    // -------------------------------------------------------------
    // ★ タイル間の pts を揃える（順再生のまま。シークはしない）
    //
    //   揃わなかった場合もここでは何もしない。前フレームを保持したまま
    //   次のラウンドへ進めば、遅れているタイルだけが前進して自然に揃う。
    //   publish 側が null を弾くので、割れた絵は表示されない。
    // -------------------------------------------------------------
    if (!align_tiles_pts(pts)) {
        warn_once_in_a_while(
            QString("[nvgpudecode] タイルの pts がまだ揃っていません FrameNo=%1")
                .arg(Frame.FrameNo));
    }

    publish_decode_frame(&got_frame);
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

// ==============================================================
// ★ デコード結果の引き渡し（旧 CUDA_RGBA_to_merge）
//
//   ここではもう RGBA を作らない。
//   NVDEC が出力した NV12 タイルのポインタを VideoFrame に載せるだけ。
//   NV12→RGBA 結合は GUI スレッドが OpenGL テクスチャへ
//   surface object 経由で直接書き込む（GLWidget::uploadToGLTexture）。
//
//   ・cuvid は avcodec_receive_frame の内部でストリーム同期しているため、
//     ここでの追加のCUDA同期は不要（従来の Dummy×3 + eventSync を削除）
//   ・タイルの寿命は vd[i].hw_frames のリングが一周するまで。
//     ringSize(=12) > 同時走行フレーム数(3〜4) なので安全。
// ==============================================================
void nvgpudecode::publish_decode_frame(const std::vector<bool>* got){

    // -------------------------------------------------------------
    // ★ 1 タイルでもフレームを取れていなければ、新しい Frame を作らない。
    //
    //   hw_frames[ringNo] が埋まっていないタイルの data[0] は null で、
    //   そのまま GUI へ渡すと「入力データが null です」になる。
    //   前回の Frame を pause 扱いで再送し、ハンドシェイクだけ回す
    //   （emit しないと receve_decode_flag が来ず、再生が止まる）。
    // -------------------------------------------------------------
    if (got) {
        for (int i = 0; i < (int)vd.size() && i < (int)got->size(); i++) {
            if ((*got)[i]) continue;

            warn_once_in_a_while(
                QString("[nvgpudecode] tile%1 のフレームを取得できませんでした。"
                        "前フレームを再送します FrameNo=%2").arg(i).arg(Frame.FrameNo));

            if (Frame.d_y[0] != nullptr)
                emit send_decode_image(Frame, true, video_reverse_flag);
            return;
        }
    }

    // ★ 念のためポインタを直接確認する（上の got と二重だが、
    //   got を渡さない経路からの呼び出しもあるため）
    for (int i = 0; i < (int)vd.size() && i < VideoFrame::MAX_TILES; i++) {
        if (vd[i].hw_frames[ringNo]->data[0] == nullptr) {
            warn_once_in_a_while(
                QString("[nvgpudecode] tile%1 の中身が空です。前フレームを再送します")
                    .arg(i));
            if (Frame.d_y[0] != nullptr)
                emit send_decode_image(Frame, true, video_reverse_flag);
            return;
        }
    }

    // ★ タイル間の pts がずれていないか最終確認（保険）
    check_tile_pts_mismatch();

    Frame.src_mode   = SRC_NV12_TILE;
    Frame.tile_count = (int)vd.size();
    Frame.tile_w     = VideoInfo.width;
    Frame.tile_h     = VideoInfo.height;

    for (int i = 0; i < (int)vd.size() && i < VideoFrame::MAX_TILES; i++) {
        AVFrame* f = vd[i].hw_frames[ringNo];
        Frame.d_y[i]      = f->data[0];
        Frame.y_pitch[i]  = (size_t)f->linesize[0];
        Frame.d_uv[i]     = f->data[1];
        Frame.uv_pitch[i] = (size_t)f->linesize[1];
    }

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

    //リングを回す（次のデコードは別の面へ）
    ringNo++;
    if(ringNo>=ringSize) ringNo = 0;

    // ★ 追い越し検出：GUIが消費する前にリングを一周していないか
    {
        const int n = ++inflight_frames;
        if (n >= ringSize) {
            qWarning() << "[nvgpudecode] RING OVERRUN inflight =" << n
                       << "/" << ringSize << " FrameNo =" << Frame.FrameNo;
        }
    }

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
                // ★ EAGAIN でパケットを捨てない
                int r = avcodec_send_packet(vd[0].codec_ctx, packet);
                if (r == AVERROR(EAGAIN)) {
                    if (avcodec_receive_frame(vd[0].codec_ctx,
                                              vd[0].hw_frames[ringNo]) == 0) {
                        av_packet_unref(packet);
                        break;
                    }
                    r = avcodec_send_packet(vd[0].codec_ctx, packet);   // ★ 再送
                }
                if (r == 0) {
                    if (avcodec_receive_frame(vd[0].codec_ctx,
                                              vd[0].hw_frames[ringNo]) == 0) {
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
            publish_decode_frame();
            break;
        }
        av_packet_unref(packet);
    }
}

//高精度シークマルチストリーム
void nvgpudecode::high_res_seek_frame_multi(int targetFrameNo){
    const int N = (int)vd.size();
    if ((int)pending_.size() != N) pending_.resize(N);

    // ----------- シーク処理 -----------
    // ビデオ/オーディオの両方 flush
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    for (auto& dec : vd) {
        avcodec_flush_buffers(dec.codec_ctx);
    }
    clear_pending();              // ★ flush したら保留パケットも捨てる

    av_seek_frame(fmt_ctx, vd[0].stream_index,
                  targetFrameNo * VideoInfo.pts_per_frame,
                  AVSEEK_FLAG_BACKWARD);

    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }
    while(!thread_stop_flag){
        std::vector<bool>    got_frame(N, false);
        std::vector<int64_t> pts(N, AV_NOPTS_VALUE);
        int got_count = 0;

        bool eof = false;
        int  read_guard = 0;
        while (got_count < N && read_guard < kLoopGuard) {
            for (int i = 0; i < N; i++) {
                int spin = 0;
                while (!got_frame[i] && !pending_[i].empty() && spin < kLoopGuard) {
                    pump_stream(i, got_frame, pts, got_count);
                    spin++;
                }
            }
            if (got_count >= N) break;
            if (!read_into_pending()) { eof = true; break; }
            read_guard++;
        }

        if (eof) {
            for (int i = 0; i < N; i++) {
                avcodec_send_packet(vd[i].codec_ctx, nullptr);
                if (avcodec_receive_frame(vd[i].codec_ctx, vd[i].hw_frames[ringNo]) == 0) {
                    pts[i] = vd[i].hw_frames[ringNo]->best_effort_timestamp;
                    if (!got_frame[i]) { got_frame[i] = true; got_count++; }
                }
            }
        }

        // ★ シーク直後はタイル間のずれが一番出やすい。ここで必ず揃える。
        //   flush 直後なので各デコーダは空から始まっており、通常は揃う。
        align_tiles_pts(pts);

        //フレーム番号記憶
        back1FrameNo = FrameNo;
        FrameNo = vd[0].hw_frames[ringNo]->best_effort_timestamp / VideoInfo.pts_per_frame;

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo<=FrameNo){
            publish_decode_frame(&got_frame);
            break;
        }

        if (eof) break;           // これ以上進めない
    }
}
