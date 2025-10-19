#include "cuda_imageprocess.h"

extern "C"{
    __global__ void nv12_to_rgba_kernel(uchar4* rgba, int rgba_pitch,const uint8_t* y_plane, int y_pitch,const uint8_t* uv_plane, int uv_pitch,int width, int height);
    __global__ void gradetion_kernel(uchar4* output_rgba, int output_rgba_step,const uchar4* input_rgba, int input_rgba_step,int width, int height);
    __global__ void flip_rgba_to_nv12_kernel(uint8_t* y_plane, int y_step,uint8_t* uv_plane, int uv_step,const uchar4* rgba, int rgba_step,int width, int height);
}

CUDA_ImageProcess::CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Constructor called";
}

CUDA_ImageProcess::~CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Destructor called";
}

bool CUDA_ImageProcess::Flip_RGBA_to_NV12(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width)
{
    void* args[] = {&d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &d_rgba, &pitch_rgba,
                    &width, &height };

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)flip_rgba_to_nv12_kernel,
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

bool CUDA_ImageProcess::NV12_to_RGBA(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width)
{
    void* args[] = {&d_rgba, &pitch_rgba,
                    &d_y, &pitch_y,
                    &d_uv, &pitch_uv,
                    &width, &height };

    dim3 block(32,32);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    cudaError_t err = cudaLaunchKernel((const void*)nv12_to_rgba_kernel,
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


