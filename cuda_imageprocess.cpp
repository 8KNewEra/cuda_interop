#include "cuda_imageprocess.h"

extern "C"{
    __global__ void nv12_to_rgba_kernel(uint8_t* rgba, int rgba_pitch,const uint8_t* y_plane, int y_pitch,const uint8_t* uv_plane, int uv_pitch,int width, int height);
    __global__ void gradetion_kernel(uint8_t* output_rgba, int output_rgba_step,const uint8_t* input_rgba, int input_rgba_step,int width, int height);
    __global__ void flip_rgba_to_nv12_kernel(uint8_t* y_plane, int y_step,uint8_t* uv_plane, int uv_step,const uint8_t* rgba, int rgba_step,int width, int height);
    __global__ void histogram_kernel(
        unsigned int* hist_r, unsigned int* hist_g, unsigned int* hist_b,
        unsigned int* global_max_r, unsigned int* global_max_g, unsigned int* global_max_b,
        cudaTextureObject_t texObj, int width, int height);
    //__global__ void draw_histogram_kernel(cudaSurfaceObject_t surface,int width,int height,const unsigned int* hist_r,const unsigned int* hist_g,const unsigned int* hist_b);
    __global__ void vbo_hist_kernel(float* vbo,int num_bins,const unsigned int* hist,unsigned int max_val);
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

bool CUDA_ImageProcess::calc_histgram(uint32_t* d_hist_r,uint32_t* d_hist_g,uint32_t* d_hist_b,unsigned int* d_max_r,unsigned int* d_max_g,unsigned int* d_max_b,cudaTextureObject_t texObj,int width,int height)
{
    void* args[] = {
        &d_hist_r,
        &d_hist_g,
        &d_hist_b,
        &d_max_r,
        &d_max_g,
        &d_max_b,
        &texObj,
        &width,
        &height
    };

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaError_t err = cudaLaunchKernel(
        (void*)histogram_kernel,
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

bool CUDA_ImageProcess::copy_histgram_vbo(float* vbo,int num_bins,const unsigned int* hist_r,const unsigned int* hist_g,const unsigned int* hist_b,unsigned int* d_max_r,unsigned int* d_max_g,unsigned int* d_max_b){
    void* args[] = {
        &vbo,
        &num_bins,
        &hist_r,
        &hist_g,
        &hist_b,
        &d_max_r,
        &d_max_g,
        &d_max_b
    };

    dim3 block(256);
    dim3 grid(1);

    cudaError_t err = cudaLaunchKernel(
        (void*)vbo_hist_kernel,
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





