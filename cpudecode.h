#ifndef CPUDECODE_H
#define CPUDECODE_H
#include "decode_thread.h"

class cpudecode:public decode_thread
{
    Q_OBJECT
public:
    using decode_thread::decode_thread;

protected:
    bool initialized_ffmpeg() override;
    const char*selectDecoder(const char* codec_name) override;
    double getFrameRate(AVFormatContext* fmt_ctx, int video_stream_index)override;
    bool get_last_frame_pts()override;
    void get_decode_audio(AVPacket* pkt)override;
    void get_decode_image()override;
    void gpu_upload();
};

#endif // CPUDECODE_H
