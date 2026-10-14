#ifndef YUV_PLANE_RING_H
#define YUV_PLANE_RING_H

#include <cuda_runtime.h>
#include <QString>
#include <QDebug>
#include "src/main/__global__.h"

// ==============================================================
// YUVプレーンリング（CPUデコード系のゼロコピー用）
//
//   ゼロコピー化により YUV→RGBA 変換はデコードスレッドではなく
//   GUIスレッド（glwidget）で走る。つまりプレーンの「消費者」が
//   別スレッドに移ったので、GUIスレッドが前フレームを変換している
//   最中にデコードスレッドが次フレームを cudaMemcpy2D で上書きしうる。
//   → nvgpudecode の hw_frames と同じく多面化して回避する。
//
//   NVDEC経路（nvgpudecode）は FFmpeg 側のフレームプールをそのまま
//   指すのでこれを必要としない。よって decode_thread には置かず、
//   必要な派生クラス（cpudecode / avidecode）が持つ。
//
//   面数は「同時に走りうるフレーム数 + 1」あれば足りる。
//   16K 422P10 は1面あたり約531MBなので増やしすぎないこと。
// ==============================================================
class YuvPlaneRing
{
public:
    static const int RING_SIZE = 3;

    YuvPlaneRing() = default;
    ~YuvPlaneRing() { release(); }

    // コピー禁止（デバイスメモリの二重解放防止）
    YuvPlaneRing(const YuvPlaneRing&)            = delete;
    YuvPlaneRing& operator=(const YuvPlaneRing&) = delete;

    // --------------------------------------------------------------
    // 確保
    //   widthBytes[] / heights[] は nplanes 個ぶん指定する
    //   packed フォーマット（UYVY422 など）は nplanes = 1
    // --------------------------------------------------------------
    bool alloc(const size_t widthBytes[3], const int heights[3],
               int nplanes, QString& error)
    {
        release();

        if (nplanes < 1 || nplanes > 3) {
            error = QString("YuvPlaneRing::alloc: invalid nplanes: %1").arg(nplanes);
            return false;
        }

        double mb_per_face = 0.0;

        for (int i = 0; i < RING_SIZE; i++) {
            for (int p = 0; p < nplanes; p++) {
                cudaError_t err = cudaMallocPitch(
                    &slots_[i].p[p],
                    &slots_[i].pitch[p],
                    widthBytes[p],
                    (size_t)heights[p]
                    );

                if (err != cudaSuccess) {
                    error = QString("cudaMallocPitch(YuvPlaneRing[%1].p[%2]) failed: %3")
                                .arg(i).arg(p)
                                .arg(QString::fromUtf8(cudaGetErrorString(err)));
                    release();
                    return false;
                }

                if (i == 0)
                    mb_per_face += (double)slots_[0].pitch[p] * heights[p] / (1024.0*1024.0);
            }
        }

        planeCount_ = nplanes;
        ringNo_     = 0;

        qDebug() << "[YuvPlaneRing]" << RING_SIZE << "faces,"
                 << nplanes << "planes," << mb_per_face << "MB/face";

        return true;
    }

    //解放
    void release()
    {
        for (int i = 0; i < RING_SIZE; i++) {
            for (int p = 0; p < 3; p++) {
                if (slots_[i].p[p]) {
                    cudaError_t err = cudaFree(slots_[i].p[p]);
                    if (err != cudaSuccess)
                        qWarning() << "[YuvPlaneRing] cudaFree failed:"
                                   << cudaGetErrorString(err);
                    slots_[i].p[p] = nullptr;
                }
                slots_[i].pitch[p] = 0;
            }
        }
        ringNo_     = 0;
        planeCount_ = 0;
    }

    // --------------------------------------------------------------
    // 現在の面を Frame に割り当てる（★cudaMemcpy2D より前に呼ぶ）
    // --------------------------------------------------------------
    void select(VideoFrame& Frame) const
    {
        for (int p = 0; p < 3; p++) {
            Frame.d_plane[p]     = slots_[ringNo_].p[p];
            Frame.plane_pitch[p] = slots_[ringNo_].pitch[p];
        }
    }

    // --------------------------------------------------------------
    // 次の面へ（★emit の後に呼ぶ）
    //   emit 済みのフレームは今の面を指しているので、
    //   ここで進めておかないと次フレームが同じ面を踏む。
    // --------------------------------------------------------------
    void advance()
    {
        ringNo_++;
        if (ringNo_ >= RING_SIZE) ringNo_ = 0;
    }

    int  planes() const { return planeCount_; }
    bool valid()  const { return planeCount_ > 0; }

private:
    struct Slot {
        uint8_t* p    [3] = {};
        size_t   pitch[3] = {};
    };

    Slot slots_[RING_SIZE];
    int  ringNo_     = 0;
    int  planeCount_ = 0;
};

#endif // YUV_PLANE_RING_H
