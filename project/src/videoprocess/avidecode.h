#ifndef AVIDECODE_H
#define AVIDECODE_H
#include "src/videoprocess/decode_thread.h"
#include "src/videoprocess/yuv_plane_ring.h"

class avidecode:public decode_thread
{
    Q_OBJECT
public:
    using decode_thread::decode_thread;

protected:
    bool initialized_ffmpeg() override;
    void get_decode_image()override;
    void high_res_seek_frame(int targetFrameNo,bool heavy_UI_flag)override;
    void gpu_upload();

    // ★ CPUデコード結果のGPU面。
    //   変換はGUIスレッドで走るので、上書き防止のため多面化する。
    //   NVDEC経路は不要なので decode_thread ではなくここで持つ。
    YuvPlaneRing planeRing;
};

#endif // AVIDECODE_H
