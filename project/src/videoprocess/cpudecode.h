#ifndef CPUDECODE_H
#define CPUDECODE_H
#include "src/videoprocess/decode_thread.h"
#include "src/videoprocess/yuv_plane_ring.h"

class cpudecode:public decode_thread
{
    Q_OBJECT
public:
    using decode_thread::decode_thread;

protected:
    bool initialized_ffmpeg() override;
    void get_decode_image()override;
    void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)override;
    const char*selectDecoder(const char* codec_name);
    void gpu_upload();

    // ★ CPUデコード結果のGPU面。
    //   変換はGUIスレッドで走るので、上書き防止のため多面化する。
    //   NVDEC経路は不要なので decode_thread ではなくここで持つ。
    YuvPlaneRing planeRing;
};

#endif // CPUDECODE_H
