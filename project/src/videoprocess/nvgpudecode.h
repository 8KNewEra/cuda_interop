#ifndef NVGPUDECODE_H
#define NVGPUDECODE_H

#include "src/videoprocess/decode_thread.h"

class nvgpudecode:public decode_thread
{
    Q_OBJECT
public:
    using decode_thread::decode_thread;

protected:
    bool initialized_ffmpeg() override;
    void get_decode_image()override;
    const char*selectDecoder(const char* codec_name);
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index);
    bool get_last_frame_pts();
    void get_singledecode_image();
    void get_multidecode_image();
    void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)override;
    void get_decode_audio();

    // ★ 旧 CUDA_RGBA_to_merge()
    //   ゼロコピー化により、ここでは RGBA を作らない。
    //   NVDEC が出力した NV12 タイルのポインタを VideoFrame に載せて
    //   GUI スレッドへ渡すだけ。実際の NV12→RGBA 結合は GUI スレッドが
    //   OpenGL テクスチャへ直接書き込む形で行う。
    void publish_decode_frame();

    void high_res_seek_frame_single(int FrameNo);
    void high_res_seek_frame_multi(int FrameNo);

    int a=0;
};

#endif // NVGPUDECODE_H
