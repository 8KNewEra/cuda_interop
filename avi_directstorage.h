#ifndef AVI_DIRECTSTORAGE_H
#define AVI_DIRECTSTORAGE_H
#include "decode_thread.h"
#include "dstorage.h"
#include <wrl/client.h>

class avi_directstorage:public decode_thread
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
    void get_decode_audio();
    void gpu_upload();

    bool initialized_dx_storage();
    void wait_for_ds_complete();
    bool read_frame_to_gpu(int frame_index,ID3D12Resource* dst);

    // ---------- DirectStorage ----------
    std::vector<int64_t> frame_offsets;
    int bytes_per_frame = 0;
    Microsoft::WRL::ComPtr<IDStorageFactory> dsFactory;
    Microsoft::WRL::ComPtr<IDStorageFile>    dsFile;
    Microsoft::WRL::ComPtr<IDStorageQueue>   dsQueue;
    Microsoft::WRL::ComPtr<ID3D12Device>     d3d12Device;
};

#endif // AVI_DIRECTSTORAGE_H
