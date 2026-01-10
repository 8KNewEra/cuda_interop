#ifndef AVI_DIRECTSTORAGE_H
#define AVI_DIRECTSTORAGE_H
#include "decode_thread.h"
#include "dstorage.h"
#include <d3d12.h>
#include <dxgi1_4.h>   // ★これ
#include <wrl.h>

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

    bool create_device();
    bool initialized_dx_storage();
    bool init_fence();
    bool create_D3D12_resource();
    bool wait_for_ds_complete();
    bool read_frame_to_gpu(int frame_index,ID3D12Resource* dst);

    // ---------- DirectStorage ----------
    std::vector<int64_t> frame_offsets;
    int bytes_per_frame = 0;
    Microsoft::WRL::ComPtr<IDStorageFactory> dsFactory=nullptr;
    Microsoft::WRL::ComPtr<IDStorageFile>    dsFile=nullptr;
    Microsoft::WRL::ComPtr<IDStorageQueue>   dsQueue=nullptr;
    Microsoft::WRL::ComPtr<ID3D12Device>     d3d12Device=nullptr;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence=nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> dstBuffer=nullptr;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;
    int count=0;

    bool init_cuda_D3D12_interop(
        ID3D12Device* device,
        ID3D12Resource* dst,
        size_t bufferSize
        );
    bool init_cuda_fence_interop(ID3D12Fence* fence);

    void* cudaPtr = nullptr;
    HANDLE sharedHandle = nullptr;
    cudaExternalMemory_t extMem = nullptr;
    size_t cudaBufferSize = 0;
    cudaExternalSemaphore_t cudaFence = nullptr;

    bool init_graph_que();
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> gfxQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdAlloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList;
    Microsoft::WRL::ComPtr<ID3D12Fence> gfxFence;
    HANDLE gfxFenceEvent;
    UINT64 gfxFenceValue = 1;
    // gfxFence 用 CUDA semaphore
    cudaExternalSemaphore_t cudaGfxFence;



};

#endif // AVI_DIRECTSTORAGE_H
