#include "cuda_imageprocess.h"
#include "qdebug.h"

CUDA_ImageProcess::CUDA_ImageProcess(){
    rgbToNv12Kernel.ptxPath = "../../cuda_kernels/rgbatonv12.ptx";
    rgbToNv12Kernel.kernelName = "rgba_to_nv12_kernel";

    nv12ToBgrKernel.ptxPath = "../../cuda_kernels/nv12tobgr.ptx";
    nv12ToBgrKernel.kernelName = "nv12_to_bgr_kernel";

    gradationKernel.ptxPath = "../../cuda_kernels/gradation.ptx";
    gradationKernel.kernelName = "gradetion_kernel";

    qDebug() << "CUDA_ImageProces: Constructor called";
}

CUDA_ImageProcess::~CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Destructor called";
}

bool CUDA_ImageProcess::RGBA_to_NV12(cv::cuda::GpuMat &rgba_gpu,cv::cuda::GpuMat &gpu_y,cv::cuda::GpuMat &gpu_uv,int height,int width){
    if (!rgbToNv12Kernel.function) {
        if(load_CUDA_Kernel(rgbToNv12Kernel)) return false;
    };

    CUdeviceptr d_rgba = reinterpret_cast<CUdeviceptr>(rgba_gpu.ptr());
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(gpu_y.ptr());
    CUdeviceptr d_uv = reinterpret_cast<CUdeviceptr>(gpu_uv.ptr());

    // カーネル引数
    void* args[] = {
        &d_rgba, &rgba_gpu.step,
        &d_y, &gpu_y.step,
        &d_uv, &gpu_uv.step,
        &width, &height
    };

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    CUresult res = cuLaunchKernel(
        rgbToNv12Kernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, 0,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }
    cuCtxSynchronize();  // or cudaDeviceSynchronize();
    return true;
}

bool CUDA_ImageProcess::NV12_to_BGR(cv::cuda::GpuMat &gpu_y,cv::cuda::GpuMat &gpu_uv,cv::cuda::GpuMat &bgr_image,int height,int width) {
    if (!nv12ToBgrKernel.function) {
        if(!load_CUDA_Kernel(nv12ToBgrKernel)) return false;
    };

    CUdeviceptr d_bgr = reinterpret_cast<CUdeviceptr>(bgr_image.ptr());
    CUdeviceptr d_y = reinterpret_cast<CUdeviceptr>(gpu_y.ptr());
    CUdeviceptr d_uv = reinterpret_cast<CUdeviceptr>(gpu_uv.ptr());

    // カーネル引数
    void* args[] = {
        &d_y, &gpu_y.step,
        &d_uv, &gpu_uv.step,
        &d_bgr, &bgr_image.step,
        &width, &height
    };

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    CUresult res = cuLaunchKernel(
        nv12ToBgrKernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, 0,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }
    cuCtxSynchronize();  // or cudaDeviceSynchronize();
    return true;
}

bool CUDA_ImageProcess::Gradation(cv::cuda::GpuMat &output,cv::cuda::GpuMat &input,int height,int width) {
    if (! gradationKernel.function) {
        if(!load_CUDA_Kernel(gradationKernel)) return false;
    };

    CUdeviceptr d_output = reinterpret_cast<CUdeviceptr>(output.ptr());
    CUdeviceptr d_input = reinterpret_cast<CUdeviceptr>(input.ptr());

    // カーネル引数
    void* args[] = {
        &d_output, &output.step,
        &d_input, &input.step,
        &width, &height
    };

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    CUresult res = cuLaunchKernel(
        gradationKernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, 0,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }
    cuCtxSynchronize();  // or cudaDeviceSynchronize();
    return true;
}

bool CUDA_ImageProcess::load_CUDA_Kernel(CudaKernelModule& kernelModule) {
    if (kernelModule.function) return true;

    // 既存コンテキストの取得
    CUcontext ctx;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        qWarning() << "CUDA context が取得できません";
        return false;
    }

    // PTXファイル読み込み
    QFile file(kernelModule.ptxPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "PTXファイルが開けません";
        return false;
    }
    QByteArray ptx = file.readAll();

    // モジュールロード
    res = cuModuleLoadDataEx(&kernelModule.module, ptx.constData(), 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        qWarning() << "cuModuleLoadDataEx失敗";
        return false;
    }

    // 関数取得
    res = cuModuleGetFunction(&kernelModule.function, kernelModule.module, kernelModule.kernelName.toUtf8().constData());
    if (res != CUDA_SUCCESS) {
        qWarning() << "cuModuleGetFunction失敗";
        return false;
    }

    return true;
}


