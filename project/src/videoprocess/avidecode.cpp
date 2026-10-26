#include "src/videoprocess/avidecode.h"


// ===============================
// avi CPUデコード
// ===============================

// ffmpeg初期化（CPU）
bool avidecode::initialized_ffmpeg()
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
    // デコーダ作成（CPU）
    // ------------------------
    {
        VideoDecorder dec{};

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
        AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;

        // ★ decoder を必ず取得する
        dec.decoder = avcodec_find_decoder(codecpar->codec_id);
        if (!dec.decoder) {
            Error_String = "rawvideo decoder not found";
            return false;
        }

        // ★ 情報オーバーレイの「Decorder:」表示用。
        //   設定しないと前に開いたファイルの値が残ったままになる。
        VideoInfo.Codec = dec.decoder->name;

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

        // ★ rawvideo は pix_fmt を上書きせず container の値を尊重する
        dec.codec_ctx->thread_count = 0;          // auto
        dec.codec_ctx->thread_type  = FF_THREAD_FRAME;
        dec.codec_ctx->pkt_timebase =fmt_ctx->streams[video_stream_index]->time_base;

        ret = avcodec_open2(dec.codec_ctx, dec.decoder, nullptr);
        if (ret < 0) {
            Error_String = QString("avcodec_open2 failed: %1")
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
    size_t planeWidthBytes[3] = {};
    int    planeHeights[3]    = {};
    int    nplanes            = 0;

    AVCodecParameters* par = fmt_ctx->streams[video_stream_index]->codecpar;

    if(par->format==AV_PIX_FMT_UYVY422){
        VideoInfo.bitdepth = 8;
        Frame.plane_format = YUVFMT_UYVY422_8;
        nplanes = 1;                                  // packed
        planeWidthBytes[0] = (size_t)VideoInfo.width * 2;
        planeHeights[0]    = VideoInfo.height;
    }else if(par->format==AV_PIX_FMT_YUV420P){
        VideoInfo.bitdepth = 8;
        Frame.plane_format = YUVFMT_YUV420P_8;
        nplanes = 3;
        planeWidthBytes[0] = (size_t)VideoInfo.width;
        planeWidthBytes[1] = (size_t)VideoInfo.width / 2;
        planeWidthBytes[2] = (size_t)VideoInfo.width / 2;
        planeHeights[0]    = VideoInfo.height;
        planeHeights[1]    = VideoInfo.height / 2;
        planeHeights[2]    = VideoInfo.height / 2;
    }
    else if(par->format==AV_PIX_FMT_YUV420P10){
        VideoInfo.bitdepth = 10;
        Frame.plane_format = YUVFMT_YUV420P_10;
        nplanes = 3;
        planeWidthBytes[0] = (size_t)VideoInfo.width * 2;
        planeWidthBytes[1] = (size_t)(VideoInfo.width / 2) * 2;
        planeWidthBytes[2] = (size_t)(VideoInfo.width / 2) * 2;
        planeHeights[0]    = VideoInfo.height;
        planeHeights[1]    = VideoInfo.height / 2;
        planeHeights[2]    = VideoInfo.height / 2;
    }else if(par->format==AV_PIX_FMT_YUV422P){
        VideoInfo.bitdepth = 8;
        Frame.plane_format = YUVFMT_YUV422P_8;
        nplanes = 3;
        planeWidthBytes[0] = (size_t)VideoInfo.width;
        planeWidthBytes[1] = (size_t)VideoInfo.width / 2;
        planeWidthBytes[2] = (size_t)VideoInfo.width / 2;
        planeHeights[0]    = VideoInfo.height;
        planeHeights[1]    = VideoInfo.height;
        planeHeights[2]    = VideoInfo.height;
    }else if(par->format==AV_PIX_FMT_YUV422P10){
        VideoInfo.bitdepth = 10;
        Frame.plane_format = YUVFMT_YUV422P_10;
        nplanes = 3;
        planeWidthBytes[0] = (size_t)VideoInfo.width * 2;
        planeWidthBytes[1] = (size_t)(VideoInfo.width / 2) * 2;
        planeWidthBytes[2] = (size_t)(VideoInfo.width / 2) * 2;
        planeHeights[0]    = VideoInfo.height;
        planeHeights[1]    = VideoInfo.height;
        planeHeights[2]    = VideoInfo.height;
    }else if(par->format==AV_PIX_FMT_GBRP10){
        VideoInfo.bitdepth = 10;
        Frame.plane_format = YUVFMT_GBRP_10;
        nplanes = 3;                                  // plane[0]=G, [1]=B, [2]=R
        for (int p = 0; p < 3; p++) {
            planeWidthBytes[p] = (size_t)VideoInfo.width * 2;
            planeHeights[p]    = VideoInfo.height;
        }
    }else{
        Error_String="非対応フォーマット";
        return false;
    }

    if (!planeRing.alloc(planeWidthBytes, planeHeights, nplanes, Error_String))
        return false;

    Frame.src_mode    = SRC_YUV_PLANE;
    Frame.plane_is_be = 0;

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
        // ★ float のまま配る。S16 へ先に落とすと 0dBFS 超えがここで潰れ、
        //   音量を下げても歪みが残る（audio_decode::apply_volume 参照）。
        VideoInfo.out_format      = AV_SAMPLE_FMT_FLT;
        av_channel_layout_default(&out_ch_layout, 2);

        // ★ チャンネル数はここで入れること。
        //   以前は get_decode_audio() の中（= 最初の音声パケットを変換した
        //   時点）でしか代入しておらず、しかもそれはエンコード中しか走らない。
        //   save_encode の init_audio_encoder() はエンコーダ構築時に
        //   VideoInfo.audio_channels を読むので、初期値の 1 を拾って
        //   モノラルで開いていた。レイアウトが決まるこの場所が正しい。
        VideoInfo.audio_channels  = out_ch_layout.nb_channels;

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

//映像ストリーム取得
void avidecode::get_decode_image(){
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

//GPUへアップロード（★ゼロコピー：RGBA化はGUIスレッドで行う）
void avidecode::gpu_upload(){
    AVFrame* f = vd[0].hw_frames[ringNo];

    // ★ 書き込む前にリングのslotを確定させる。
    //   emit 済みの前フレームは別の面を指しているので上書きされない。
    planeRing.select(Frame);

    // GPUアップロード
    //   cudaMemcpy2D(同期版) はホストに対して同期なので、
    //   戻った時点でデータはデバイス上にある。
    //   → GUIスレッドが後から起動するカーネルからは必ず見える。
    //   従来あったダミーカーネル + イベント同期は不要なので削除した。
    //
    //   ★ GBRP10 は data[0]=G, data[1]=B, data[2]=R の順で入っている。
    //      プレーンもその並びのまま格納し、順序の入れ替えは
    //      呼び出し側(glwidget)で行う。
    const int W = VideoInfo.width;
    const int H = VideoInfo.height;

    switch (f->format) {
    case AV_PIX_FMT_UYVY422:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0],
                     W * 2, H, cudaMemcpyHostToDevice);
        Frame.plane_format = YUVFMT_UYVY422_8;
        Frame.plane_is_be  = 0;
        break;

    case AV_PIX_FMT_YUV420P:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0], W,     H,     cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                     f->data[1], f->linesize[1], W / 2, H / 2, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                     f->data[2], f->linesize[2], W / 2, H / 2, cudaMemcpyHostToDevice);
        Frame.plane_format = YUVFMT_YUV420P_8;
        Frame.plane_is_be  = 0;
        break;

    case AV_PIX_FMT_YUV420P10:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0], W * 2,       H,     cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                     f->data[1], f->linesize[1], (W / 2) * 2, H / 2, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                     f->data[2], f->linesize[2], (W / 2) * 2, H / 2, cudaMemcpyHostToDevice);
        Frame.plane_format = YUVFMT_YUV420P_10;
        Frame.plane_is_be  = 0;   // case ラベルが LE なので常に LE
        break;

    case AV_PIX_FMT_YUV422P:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0], W,     H, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                     f->data[1], f->linesize[1], W / 2, H, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                     f->data[2], f->linesize[2], W / 2, H, cudaMemcpyHostToDevice);
        Frame.plane_format = YUVFMT_YUV422P_8;
        Frame.plane_is_be  = 0;
        break;

    case AV_PIX_FMT_YUV422P10:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0], W * 2,       H, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                     f->data[1], f->linesize[1], (W / 2) * 2, H, cudaMemcpyHostToDevice);
        cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                     f->data[2], f->linesize[2], (W / 2) * 2, H, cudaMemcpyHostToDevice);
        Frame.plane_format = YUVFMT_YUV422P_10;
        Frame.plane_is_be  = 0;   // case ラベルが LE なので常に LE
        break;

    case AV_PIX_FMT_GBRP10:
        cudaMemcpy2D(Frame.d_plane[0], Frame.plane_pitch[0],
                     f->data[0], f->linesize[0], W * 2, H, cudaMemcpyHostToDevice);  // G
        cudaMemcpy2D(Frame.d_plane[1], Frame.plane_pitch[1],
                     f->data[1], f->linesize[1], W * 2, H, cudaMemcpyHostToDevice);  // B
        cudaMemcpy2D(Frame.d_plane[2], Frame.plane_pitch[2],
                     f->data[2], f->linesize[2], W * 2, H, cudaMemcpyHostToDevice);  // R
        Frame.plane_format = YUVFMT_GBRP_10;
        Frame.plane_is_be  = 0;
        break;

    default:
        // ★ ここで何も出さずに返すと emit が飛ばず、GUI からの
        //   receve_decode_flag が返ってこないので再生が止まったままになる。
        //   前のコマを送り直してハンドシェイクだけ回す。
        qWarning() << "[avidecode] unsupported pixel format:" << f->format;
        if (Frame.d_plane[0] != nullptr) {
            ++inflight_frames;
            emit send_decode_image(Frame, true, video_reverse_flag);
        }
        return;
    }

    Frame.src_mode = SRC_YUV_PLANE;

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
            qWarning() << "[avidecode] PLANE RING OVERRUN inflight =" << n
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
void avidecode::high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)
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
    if(heavy_UI_flag){
        //UIはfalse
        emit heavy_process_signal(false);
    }

    // ----------- シーク処理 -----------
    avcodec_flush_buffers(vd[0].codec_ctx);
    if (audio_ctx)
        avcodec_flush_buffers(audio_ctx);

    av_seek_frame(fmt_ctx, vd[0].stream_index,
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
            int ret = av_read_frame(fmt_ctx, packet);

            // ---------- EOF ----------
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

                av_seek_frame(fmt_ctx, vd[0].stream_index,
                              frame_to_pts((int64_t)seek_frame),
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
}
