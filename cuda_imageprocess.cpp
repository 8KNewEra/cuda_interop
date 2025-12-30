#include "cuda_imageprocess.h"

//CUDAカーネル関数
extern "C"{
    __global__ void nv12_to_rgba_8bit_kernel(uint8_t* rgba, size_t rgba_pitch,const uint8_t* y_plane, size_t y_pitch,const uint8_t* uv_plane, size_t uv_pitch,int width, int height);
    __global__ void nv12_to_rgba_10bit_kernel(uint8_t* rgba, size_t rgba_pitch,const uint8_t* y_plane, size_t y_pitch,const uint8_t* uv_plane, size_t uv_pitch,int width, int height);
    __global__ void yuv420p_to_rgba_8bit_kernel(uint8_t* rgba, size_t rgba_pitch,const uint8_t* y_plane, size_t y_pitch,const uint8_t* u_plane, size_t u_pitch,const uint8_t* v_plane, size_t v_pitch,int width, int height);
    __global__ void yuv420p_to_rgba_10bit_kernel(uint8_t* rgba, size_t rgba_pitch,const uint8_t* y_plane, size_t y_pitch,const uint8_t* u_plane, size_t u_pitch,const uint8_t* v_plane, size_t v_pitch,int width, int height);
    __global__ void gradetion_kernel(uint8_t* output_rgba, int output_rgba_step,const uint8_t* input_rgba, int input_rgba_step,int width, int height);
    __global__ void flip_rgba_to_nv12_kernel(uint8_t* y_plane, size_t y_step,uint8_t* uv_plane,size_t uv_step,const uint8_t* rgba, size_t rgba_step,int width, int height);
    __global__ void calc_histogram_shared_kernel(HistData* Histdata,cudaTextureObject_t texObj, int width, int height);
    __global__ void calc_histogram_normal_kernel(HistData* Histdata,cudaTextureObject_t texObj, int width, int height);
    __global__ void histgram_normalize_kernel(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats);
    __global__ void histgram_status_kernel(HistData* Histdata,HistStats* out_stats);
    __global__ void image_combine_x2_kernel(uint8_t* out, size_t pitchOut,const uint8_t* img1, size_t pitch1,const uint8_t* img2, size_t pitch2,int width, int height);
    __global__ void image_combine_x4_kernel(uint8_t* out, size_t pitchOut,const uint8_t* img1, size_t pitch1,const uint8_t* img2, size_t pitch2,const uint8_t* img3, size_t pitch3,const uint8_t* img4, size_t pitch4,int width, int height, int blend);
    __global__ void image_split_x4_kernel(uint8_t* Out0, size_t pitch0,uint8_t* Out1, size_t pitch1,uint8_t* Out2, size_t pitch2,uint8_t* Out3, size_t pitch3,const uint8_t* In, size_t pitchIn,int width, int height);
    __global__ void nv12x4_to_rgba_merge_kernel(
        const uint8_t* y0,  size_t pitchY0,const uint8_t* uv0, size_t pitchUV0,
        const uint8_t* y1,  size_t pitchY1,const uint8_t* uv1, size_t pitchUV1,
        const uint8_t* y2,  size_t pitchY2,const uint8_t* uv2, size_t pitchUV2,
        const uint8_t* y3,  size_t pitchY3,const uint8_t* uv3, size_t pitchUV3,
        uint8_t* out, size_t pitchOut,int outW, int outH,int srcW, int srcH);
    __global__ void rgba_to_nv12x4_flip_split_kernel(
        const uint8_t* In, size_t pitchIn,
        uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
        uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
        uint8_t* y2,  size_t pitchY2,uint8_t* uv2, size_t pitchUV2,
        uint8_t* y3,  size_t pitchY3,uint8_t* uv3, size_t pitchUV3,
        int srcW, int srcH,int outW, int outH);
    //__global__ void draw_histogram_kernel(cudaSurfaceObject_t surface,int width,int height,const unsigned int* hist_r,const unsigned int* hist_g,const unsigned int* hist_b);
}

CUDA_ImageProcess::CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Constructor called";
}

CUDA_ImageProcess::~CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Destructor called";
}

//NV12→RGBA
bool CUDA_ImageProcess::NV12_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,cudaStream_t stream)
{
    void* args[] = {&d_rgba, &pitch_rgba,
                    &d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12_to_rgba_8bit_kernel,
                                       grid, block, args, 0, stream);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::NV12_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width,cudaStream_t stream)
{
    void* args[] = {&d_rgba, &pitch_rgba,
                    &d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12_to_rgba_10bit_kernel,
                                       grid, block, args, 0, stream);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//反転→RGBA→NV12
bool CUDA_ImageProcess::Flip_RGBA_to_NV12(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width,cudaStream_t stream)
{
    void* args[] = {&d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &d_rgba, &pitch_rgba,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)flip_rgba_to_nv12_kernel,
                     grid, block, args, 0, stream);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//コントラストグラデーション
bool CUDA_ImageProcess::Gradation(uint8_t *output,size_t pitch_output,uint8_t *input,size_t pitch_input,int height,int width) {
    void* args[] = {&output, &pitch_output,
                    &input,&pitch_input,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err =cudaLaunchKernel((const void*)gradetion_kernel,
                     grid, block, args, 0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//ヒストグラム計算
bool CUDA_ImageProcess::calc_histgram(HistData* Histdata,cudaTextureObject_t texObj,int width,int height)
{
    void* args[] = {
        &Histdata,
        &texObj,
        &width,
        &height
    };

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaError_t err = cudaLaunchKernel(
        (void*)calc_histogram_shared_kernel,
        grid, block,
        args,
        0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed:" << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error:" << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//ヒストグラム詳細を計算
bool CUDA_ImageProcess::histogram_status(HistData* Histdata,HistStats* out_stats){
    void* args[] = {
        &Histdata,
        &out_stats,
    };

    dim3 block(256);
    dim3 grid(1);

    cudaError_t err = cudaLaunchKernel(
        (void*)histgram_status_kernel,
        grid, block,
        args,
        0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed:" << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error:" << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//ヒストグラム正規化→VBO
bool CUDA_ImageProcess::histgram_normalize(float* vbo,int num_bins,HistData* Histdata,HistStats* input_stats){
    void* args[] = {
        &vbo,
        &num_bins,
        &Histdata,
        &input_stats
    };

    dim3 block(256);
    dim3 grid(1);

    cudaError_t err = cudaLaunchKernel(
        (void*)histgram_normalize_kernel,
        grid, block,
        args,
        0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed:" << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error:" << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//CUDAで描画するらしい
bool CUDA_ImageProcess::draw_histgram(cudaSurfaceObject_t surfOut,int width,int height,uint32_t* d_hist_r,uint32_t* d_hist_g,uint32_t* d_hist_b,unsigned int* max_r,unsigned int* max_g,unsigned int* max_b){
    void* args[] = {
        &surfOut,
        &width,
        &height,
        &d_hist_r,
        &d_hist_g,
        &d_hist_b
    };

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // cudaError_t err = cudaLaunchKernel(
    //     (void*)draw_histogram_kernel,
    //     grid, block,
    //     args,
    //     0, nullptr);

    // if (err != cudaSuccess) {
    //     qDebug() << "cudaLaunchKernel failed:" << cudaGetErrorString(err);
    //     return false;
    // }

    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     qDebug() << "Kernel launch error:" << cudaGetErrorString(err);
    //     return false;
    // }

    return true;
}

bool CUDA_ImageProcess::image_combine_x2(uint8_t* out, size_t pitchOut,uint8_t* img1, size_t pitch1,uint8_t* img2, size_t pitch2,int width, int height){
    void* args[] = {&out, &pitchOut,
                    &img1,&pitch1,
                    &img2,&pitch2,
                    &width, &height};

    dim3 block(16, 16);
    dim3 grid((width * 2 + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaError_t err =cudaLaunchKernel((const void*)image_combine_x2_kernel,
                                       grid, block, args, 0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::image_combine_x4(uint8_t* out, size_t pitchOut,uint8_t* img1, size_t pitch1,uint8_t* img2, size_t pitch2,uint8_t* img3, size_t pitch3,uint8_t* img4, size_t pitch4,int width, int height, int blend){
    void* args[] = {&out, &pitchOut,
                    &img1,&pitch1,
                    &img2,&pitch2,
                    &img3,&pitch3,
                    &img4,&pitch4,
                    &width, &height,
                    &blend};

    dim3 block(16, 16);
    dim3 grid((width * 2 + block.x - 1) / block.x,
              (height * 2 + block.y - 1) / block.y);

    cudaError_t err =cudaLaunchKernel((const void*)image_combine_x4_kernel,
                                       grid, block, args, 0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::image_split_x4(uint8_t* Out[4], size_t pitch[4],uint8_t* In, size_t pitchIn,int width, int height,cudaStream_t stream){
    width = width*1/2;
    height = height*1/2;

    void* args[] = {&Out[2], &pitch[2],
                    &Out[3], &pitch[3],
                    &Out[0], &pitch[0],
                    &Out[1], &pitch[1],
                    &In, &pitchIn,
                    &width, &height
                    };

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
        );

    cudaError_t err =cudaLaunchKernel((const void*)image_split_x4_kernel,
                                       grid, block, args, 0, stream);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}


//4分割デコード
bool CUDA_ImageProcess::nv12x4_to_rgba_merge(uint8_t* y0,  size_t pitchY0,uint8_t* uv0, size_t pitchUV0,
                                             uint8_t* y1,  size_t pitchY1,uint8_t* uv1, size_t pitchUV1,
                                             uint8_t* y2,  size_t pitchY2,uint8_t* uv2, size_t pitchUV2,
                                             uint8_t* y3,  size_t pitchY3,uint8_t* uv3, size_t pitchUV3,
                                             uint8_t* out, size_t pitchOut,int outW, int outH,int srcW, int srcH){

    void* args[] = {&y0, &pitchY0,&uv0,&pitchUV0,
                    &y1, &pitchY1,&uv1,&pitchUV1,
                    &y2, &pitchY2,&uv2,&pitchUV2,
                    &y3, &pitchY3,&uv3,&pitchUV3,
                    &out, &pitchOut,&outW, &outH,&srcW,&srcH
    };

    dim3 block(16, 16);
    dim3 grid(
        (outW  + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y
        );

    cudaError_t err =cudaLaunchKernel((const void*)nv12x4_to_rgba_merge_kernel,
                                       grid, block, args, 0, nullptr);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

//4分割エンコード
bool CUDA_ImageProcess::rgba_to_nv12x4_flip_split(uint8_t* In, size_t pitchIn,
                                                  uint8_t* y0,  size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
                                                  uint8_t* y1,  size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
                                                  uint8_t* y2,  size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
                                                  uint8_t* y3,  size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
                                                  int srcW, int srcH, int outW, int outH, cudaStream_t stream){

    void* args[] = {&In, &pitchIn,
        &y0, &pitchY0,&uv0,&pitchUV0,
        &y1, &pitchY1,&uv1,&pitchUV1,
        &y2, &pitchY2,&uv2,&pitchUV2,
        &y3, &pitchY3,&uv3,&pitchUV3,
        &srcW, &srcH,&outW,&outH
    };

    dim3 block(16, 16);
    dim3 grid(
        (srcW  + block.x - 1) / block.x,
        (srcH + block.y - 1) / block.y
        );

    cudaError_t err =cudaLaunchKernel((const void*)rgba_to_nv12x4_flip_split_kernel,
                                       grid, block, args, 0, stream);

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::yuv420p_to_RGBA_8bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,cudaStream_t stream){
    void* args[] = {&d_rgba, &pitch_rgba,
                    &d_y, &pitch_y,
                    &d_u, &pitch_u,
                    &d_v, &pitch_v,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    cudaError_t err = cudaLaunchKernel((const void*)yuv420p_to_rgba_8bit_kernel,
                                       grid, block, args, 0, stream);

    cudaGetLastError();   // ← 必ずチェック

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::yuv420p_to_RGBA_10bit(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_u, size_t pitch_u,uint8_t* d_v, size_t pitch_v,int width, int height,cudaStream_t stream){
    void* args[] = {&d_rgba, &pitch_rgba,
                    &d_y, &pitch_y,
                    &d_u, &pitch_u,
                    &d_v, &pitch_v,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    cudaError_t err = cudaLaunchKernel((const void*)yuv420p_to_rgba_10bit_kernel,
                                       grid, block, args, 0, stream);

    cudaGetLastError();   // ← 必ずチェック

    if (err != cudaSuccess) {
        qDebug() << "cudaLaunchKernel failed: " << cudaGetErrorString(err);
        return false;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        qDebug() << "Kernel launch error: " << cudaGetErrorString(err);
        return false;
    }

    return true;
}
