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

        //リングバッファフレーム作成
        dec.hw_frames.resize(ringSize);
        for (int j = 0; j < ringSize; j++) {
            dec.hw_frames[j] = av_frame_alloc();
            if (!dec.hw_frames[j]) {
                Error_String = "av_frame_alloc failed";
                return false;
            }
        }

        dec.stream_index = video_stream_index;
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

        // CPU デコード（hw_device_ctx は alloc 時点で null）
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

    // ★ 欠番の多いファイルでは avg_frame_rate が下振れする。実測で補正する。
    fix_frame_grid(vd[0].stream_index, time_base_d);

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
    // YUVプレーンのGPU面を確保
    //   ★ RGBA 化は GUIスレッドが GLテクスチャへ直接書く（ゼロコピー）。
    //     消費者が別スレッドなので、上書き事故を防ぐため多面化する。
    // ------------------------
    const int bytesPerSample = (VideoInfo.bitdepth == 10) ? 2 : 1;

    const size_t planeWidthBytes[3] = {
        (size_t)VideoInfo.width * bytesPerSample,
        (size_t)(VideoInfo.width / 2) * bytesPerSample,
        (size_t)(VideoInfo.width / 2) * bytesPerSample
    };
    const int planeHeights[3] = {
        VideoInfo.height,
        VideoInfo.height / 2,
        VideoInfo.height / 2
    };

    if (!planeRing.alloc(planeWidthBytes, planeHeights, 3, Error_String))
        return false;

    Frame.src_mode     = SRC_YUV_PLANE;
    Frame.plane_format = (VideoInfo.bitdepth == 10) ? YUVFMT_YUV420P_10
                                                    : YUVFMT_YUV420P_8;
    Frame.plane_is_be  = 0;

    // ------------------------
    // 音声ストリーム探索
    // ------------------------
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    // 音声（エンコード入力用）。開けなければ音声なしで続ける（decode_thread::open_audio_decoder）
    open_audio_decoder();

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
    } else if (strcmp(codec_name, "hevc") == 0) {
        codec="hevc";
    }else if (strcmp(codec_name, "av1") == 0){
        codec="av1";
    }else{
        codec=nullptr;
    }
    // ★ nullptr を std::string へ代入してはいけない（未定義動作）。
    //   対応していないコーデックのときは表示名を変えずに返す。
    //   呼び出し側は戻り値の null を見てエラーにする。
    if (codec) VideoInfo.Codec = codec;
    return codec;
}

//映像ストリーム取得
void cpudecode::get_decode_image(){
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

        seek_file(vd[0].stream_index,
                      frame_to_pts(slider_No),
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
    // ★ 終端（読めない失敗も）を何回見たか。終端 → 読み直し → 終端 を永久に繰り返さない
    //   （ここは processFrame のミューテックスの中なので、回り続けると GUI まで固まる）
    int eof_rounds = 0;
    while (!thread_stop_flag) {
        int ret = read_packet();

        // ---------- EOF（読めない失敗も。そのときは read_packet が fail_decode 済みなので抜ける）----------
        if (ret < 0) {
            if (decode_failed_) { av_packet_unref(packet); return; }

            avcodec_send_packet(vd[0].codec_ctx, nullptr);

            if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                av_packet_unref(packet);
                break;
            }

            // 終端→先頭に戻ってループ（2 回目の終端は範囲の終わりと同じ扱い）
            if (Frame.FrameNo < VideoInfo.end_range_framesNo - 1 && ++eof_rounds < 2) {
                avcodec_flush_buffers(vd[0].codec_ctx);
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                seek_file(vd[0].stream_index,
                              frame_to_pts(Frame.FrameNo + 1),
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
            const int sr = avcodec_send_packet(vd[0].codec_ctx, packet);
            if (sr < 0 && sr != AVERROR(EAGAIN) && sr != AVERROR_EOF) warn_ffmpeg("avcodec_send_packet", sr);   // 壊れたパケット等。捨てて続ける
            if (sr == 0) {
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
                }
            }
        }
        av_packet_unref(packet);
    }
    if (thread_stop_flag) { av_packet_unref(packet); return; }   // 閉じる途中

    gpu_upload();
    av_packet_unref(packet);
}

//GPUへアップロード（★ゼロコピー：RGBA化はGUIスレッドで行う）
void cpudecode::gpu_upload(){
    const int bytesPerSample = (VideoInfo.bitdepth == 10) ? 2 : 1;
    AVFrame* f = vd[0].hw_frames[ringNo];

    // 取れていないコマ（終端でデコーダが空だった等）は送らない。前のコマを送り直してハンドシェイクだけ回す
    if (!f || !f->data[0]) {
        if (Frame.d_plane[0] != nullptr) {
            ++inflight_frames;
            emit send_decode_image(Frame, true, video_reverse_flag);
        }
        return;
    }

    // ★ 書き込む前にリングのslotを確定させる。
    //   emit 済みの前フレームは別の面を指しているので上書きされない。
    planeRing.select(Frame);
    cudaGetLastError();   // 前の呼び出しの残りを消す（下の転送の失敗の判定を、この転送の結果だけにする）

    // GPUアップロード
    //   cudaMemcpy2D(同期版) はホストに対して同期なので、
    //   戻った時点でデータはデバイス上にある。
    //   → GUIスレッドが後から起動するカーネルからは必ず見える。
    //   従来あったダミーカーネル + イベント同期は不要なので削除した。
    cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                 f->data[0], f->linesize[0],
                 VideoInfo.width * bytesPerSample,
                 VideoInfo.height,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                 f->data[1], f->linesize[1],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                 f->data[2], f->linesize[2],
                 (VideoInfo.width / 2) * bytesPerSample,
                 VideoInfo.height / 2,
                 cudaMemcpyHostToDevice);

    // 転送の失敗（GPU のエラー）。同期版 cudaMemcpy2D の失敗は直後の cudaGetLastError に出る。続けられないので止める
    if (const cudaError_t ce = cudaGetLastError(); ce != cudaSuccess) {
        fail_decode(QStringLiteral("GPU への転送に失敗しました（%1）").arg(QString::fromUtf8(cudaGetErrorString(ce))));
        return;
    }

    // 変換に必要な情報をフレームに載せる
    Frame.src_mode     = SRC_YUV_PLANE;
    Frame.plane_format = (VideoInfo.bitdepth == 10) ? YUVFMT_YUV420P_10
                                                    : YUVFMT_YUV420P_8;
    Frame.plane_is_be  = (f->format == AV_PIX_FMT_YUV420P10BE);

    // フレーム番号と時刻（基底で共通化。0 除算と NOPTS もそこでふさいでいる）
    advance_frame_no(vd[0].hw_frames[ringNo]->best_effort_timestamp);

    //リングを回す
    ringNo++;
    if(ringNo>=ringSize) ringNo = 0;

    // ★ 追い越し検出：GUIが消費する前に YuvPlaneRing を一周していないか
    //   （こちらの実効段数は planeRing の RING_SIZE）
    {
        const int n = ++inflight_frames;
        if (n >= YuvPlaneRing::RING_SIZE) {
            qWarning() << "[cpudecode] PLANE RING OVERRUN inflight =" << n
                       << "/" << YuvPlaneRing::RING_SIZE
                       << " FrameNo =" << Frame.FrameNo;
        }
    }

    emit send_decode_image(Frame,false,video_reverse_flag);

    // ★ emit の後に次の面へ。
    //   GUIスレッドは今 emit した面を読むので、ここで進めておかないと
    //   次フレームが同じ面を踏む。
    planeRing.advance();
}

//高精度シーク
void cpudecode::high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)
{
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
    // 重い処理が始まったらUIを止める
    if(heavy_UI_flag){
        //UIはfalse
        emit heavy_process_signal(false);
    }

    // ----------- シーク処理 -----------
    avcodec_flush_buffers(vd[0].codec_ctx);
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    seek_file(vd[0].stream_index,
                  frame_to_pts(targetFrameNo),
                  AVSEEK_FLAG_BACKWARD);

    // ----------- パケット読み込みループ -----------
    //デコードループ(最初はback1Frame記憶用)
    int FrameNo = targetFrameNo-1;
    if(FrameNo<0){
        FrameNo = 0;
    }
    // ★ 終端に当たってデコーダも空、という状態を何回見たか。
    //   目標が実在しない位置だと、読み直し -> 終端 を永久に繰り返す。
    //   ここは processFrame のミューテックスの中なので、回り続けると
    //   GUI スレッドまで巻き込んで固まる。
    int eof_rounds = 0;

    while(!thread_stop_flag){
        // ★ 内側にも停止要求を見せる（長いスキャン中にクローズされても抜ける）
        while (!thread_stop_flag) {
            int ret = read_packet();

            // ---------- EOF（読めない失敗も。read_packet が fail_decode 済み）----------
            if (ret < 0) {
                avcodec_send_packet(vd[0].codec_ctx, nullptr);
                if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                    av_packet_unref(packet);
                    break;
                }

                // ★ デコーダも空。1 回だけ読み直しを許し、それでも届かなければ
                //   取れている最後のコマを出して諦める。
                if (++eof_rounds >= 2) {
                    av_packet_unref(packet);
                    gpu_upload();
                    if (heavy_UI_flag) emit heavy_process_signal(true);   // 下の通常の終わりと同じく UI を戻す
                    return;
                }

                // 終端→先頭に戻ってループ
                uint64_t seek_frame{};
                if (Frame.FrameNo < VideoInfo.end_range_framesNo - 1) {
                    seek_frame = Frame.FrameNo + 1;
                }

                avcodec_flush_buffers(vd[0].codec_ctx);
                if (audio_ctx)
                    avcodec_flush_buffers(audio_ctx);

                seek_file(vd[0].stream_index,
                              frame_to_pts((int64_t)seek_frame),
                              AVSEEK_FLAG_ANY);

                av_packet_unref(packet);
                continue;
            }

            // ----------- VIDEO PACKET -----------
            if (packet->stream_index == vd[0].stream_index) {
                const int sr = avcodec_send_packet(vd[0].codec_ctx, packet);
                if (sr < 0 && sr != AVERROR(EAGAIN) && sr != AVERROR_EOF) warn_ffmpeg("avcodec_send_packet", sr);   // 壊れたパケット等。捨てて続ける
                if (sr == 0) {
                    if (avcodec_receive_frame(vd[0].codec_ctx, vd[0].hw_frames[ringNo]) == 0) {
                        av_packet_unref(packet);
                        break;
                    }
                }
            }
            // ★ 映像以外のパケットもここで解放する。シークは目標まで
            //   読み続けるので、取りこぼすとその区間ぶん丸ごと漏れる。
            av_packet_unref(packet);
        }

        //1 つ手前のコマを覚えてから、今のコマの番号を出す
        back1FrameNo = FrameNo;
        FrameNo = (VideoInfo.pts_per_frame > 0.0)
                      ? pts_to_frame(vd[0].hw_frames[ringNo]->best_effort_timestamp)
                      : FrameNo;

        //ターゲットフレームの番号かどうか判定
        if(targetFrameNo <= FrameNo){
            gpu_upload();
            break;
        }
        av_packet_unref(packet);
    }

    // 重い処理が終わったらUIを戻す
    if(heavy_UI_flag){
        emit heavy_process_signal(true);
    }
}

