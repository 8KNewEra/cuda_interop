#include "cuda_imageprocess.h"
#include "qdebug.h"

CUDA_ImageProcess::CUDA_ImageProcess(){
    rgbToNv12Kernel.ptxPath = "../../cuda_kernels/ptx_out/fliprgbatonv12.ptx";
    rgbToNv12Kernel.kernelName = "flip_rgba_to_nv12_kernel";

    nv12TorgbaKernel.ptxPath = "../../cuda_kernels/ptx_out/nv12torgba.ptx";
    nv12TorgbaKernel.kernelName = "nv12_to_rgba_kernel";

    gradationKernel.ptxPath = "../../cuda_kernels/ptx_out/gradation.ptx";
    gradationKernel.kernelName = "gradetion_kernel";

    qDebug() << "CUDA_ImageProces: Constructor called";
}

CUDA_ImageProcess::~CUDA_ImageProcess(){
    qDebug() << "CUDA_ImageProces: Destructor called";
}

bool CUDA_ImageProcess::initGraph() {
    cuGraphCreate(&g_graph.graph, 0);

    qDebug()<<"aaaaaaaaaaaaaa";
    load_CUDA_Kernel(nv12TorgbaKernel);
    load_CUDA_Kernel(gradationKernel);

    {
        CUdeviceptr dummy = 0;
        int step = 0;
        int w = 1, h = 1;
        void* args[] = { &dummy, &step, &dummy, &step, &dummy, &step, &w, &h };

        CUDA_KERNEL_NODE_PARAMS_v2 nv12Params = {};
        nv12Params.func = nv12TorgbaKernel.function;
        nv12Params.gridDimX  = 1;
        nv12Params.gridDimY  = 1;
        nv12Params.gridDimZ  = 1;
        nv12Params.blockDimX = 32;
        nv12Params.blockDimY = 32;
        nv12Params.blockDimZ = 1;
        nv12Params.kernelParams = args;

        cuGraphAddKernelNode(&g_graph.nodeNv12ToRgba, g_graph.graph,
                             nullptr, 0, &nv12Params);
    }

    {
        CUdeviceptr dummy = 0;
        int step = 0;
        int w = 1, h = 1;
        void* args[] = { &dummy, &step, &dummy, &step, &w, &h };

        CUDA_KERNEL_NODE_PARAMS_v2 gradParams = {};
        gradParams.func = gradationKernel.function;
        gradParams.gridDimX  = 1;
        gradParams.gridDimY  = 1;
        gradParams.gridDimZ  = 1;
        gradParams.blockDimX = 32;
        gradParams.blockDimY = 32;
        gradParams.blockDimZ = 1;
        gradParams.kernelParams = args;

        CUgraphNode deps[] = { g_graph.nodeNv12ToRgba };
        cuGraphAddKernelNode(&g_graph.nodeGradation, g_graph.graph,
                             deps, 1, &gradParams);
    }

    CUresult res = cuGraphInstantiateWithFlags(
        &g_graph.graphExec,
        g_graph.graph,
        0
        );

    if (res != CUDA_SUCCESS) {
        const char* name;
        const char* msg;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);
        printf("cuGraphInstantiateWithFlags failed: %s (%s)\n", name, msg);
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::RGBA_to_NV12(uint8_t* d_rgba, size_t pitch_rgba,uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width){
    if (!rgbToNv12Kernel.function) {
        if(load_CUDA_Kernel(rgbToNv12Kernel)) return false;
    };

    CUdeviceptr cu_rgba = reinterpret_cast<CUdeviceptr>(d_rgba);
    CUdeviceptr cu_y = reinterpret_cast<CUdeviceptr>(d_y);
    CUdeviceptr cu_uv = reinterpret_cast<CUdeviceptr>(d_uv);

    // カーネル引数
    void* args[] = {
        &cu_rgba, &pitch_rgba,
        &cu_y, &pitch_y,
        &cu_uv, &pitch_uv,
        &width, &height
    };

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    CUresult res = cuLaunchKernel(
        rgbToNv12Kernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, nullptr,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::NV12_to_RGBA(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,uint8_t* d_rgba, size_t pitch_rgba,int height, int width)
{
    if (!nv12TorgbaKernel.function) {
        if (!load_CUDA_Kernel(nv12TorgbaKernel)) return false;
    }

    CUdeviceptr cu_y   = reinterpret_cast<CUdeviceptr>(d_y);
    CUdeviceptr cu_uv  = reinterpret_cast<CUdeviceptr>(d_uv);
    CUdeviceptr cu_rgba = reinterpret_cast<CUdeviceptr>(d_rgba);

    int y_step   = static_cast<int>(pitch_y);
    int uv_step  = static_cast<int>(pitch_uv);
    int rgba_step = static_cast<int>(pitch_rgba);

    void* args[] = { &cu_y, &y_step, &cu_uv, &uv_step, &cu_rgba, &rgba_step, &width, &height };

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    CUresult res = cuLaunchKernel(
        nv12TorgbaKernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, nullptr,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::Gradation(uint8_t *output,size_t pitch_output,uint8_t *input,size_t pitch_input,int height,int width) {
    if (! gradationKernel.function) {
        if(!load_CUDA_Kernel(gradationKernel)) return false;
    };

    CUdeviceptr cu_output = reinterpret_cast<CUdeviceptr>(output);
    CUdeviceptr cu_input = reinterpret_cast<CUdeviceptr>(input);

    // カーネル引数
    void* args[] = {
        &cu_output, &pitch_output,
        &cu_input, &pitch_input,
        &width, &height
    };

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    CUresult res = cuLaunchKernel(
        gradationKernel.function,
        grid.x, grid.y, 1,
        block.x, block.y, 1,
        0, nullptr,
        args, nullptr
        );

    if (res != CUDA_SUCCESS) {
        qWarning() << "カーネル起動失敗";
        return false;
    }

    return true;
}

bool CUDA_ImageProcess::runGraph(uint8_t* d_y, size_t pitch_y,
                                 uint8_t* d_uv, size_t pitch_uv,
                                 uint8_t* d_rgba, size_t pitch_rgba,
                                 uint8_t* d_output, size_t pitch_output,
                                 int width, int height, CUstream stream) {
    if (!g_graph.graphExec) {
        qWarning() << "GraphExec not initialized";
        return false;
    }

    CUresult res;

    {
        CUdeviceptr cu_y     = (CUdeviceptr)d_y;
        CUdeviceptr cu_uv    = (CUdeviceptr)d_uv;
        CUdeviceptr cu_rgba  = (CUdeviceptr)d_rgba;
        int y_step   = static_cast<int>(pitch_y);
        int uv_step  = static_cast<int>(pitch_uv);
        int rgba_step= static_cast<int>(pitch_rgba);

        void* args[] = { &cu_y, &y_step, &cu_uv, &uv_step, &cu_rgba, &rgba_step, &width, &height };

        CUDA_KERNEL_NODE_PARAMS_v2 params = {};
        params.func = nv12TorgbaKernel.function;
        params.kernelParams = args;
        params.extra = nullptr;
        params.gridDimX = (width + 31) / 32;
        params.gridDimY = (height + 31) / 32;
        params.gridDimZ = 1;
        params.blockDimX = 32;
        params.blockDimY = 32;
        params.blockDimZ = 1;

        res = cuGraphExecKernelNodeSetParams(g_graph.graphExec, g_graph.nodeNv12ToRgba, &params);
        if (res != CUDA_SUCCESS) { qWarning() << "cuGraphExecKernelNodeSetParams nv12->rgba failed" << res; return false; }
    }

    {
        CUdeviceptr cu_output = (CUdeviceptr)d_output;
        CUdeviceptr cu_input  = (CUdeviceptr)d_rgba;
        int output_step = static_cast<int>(pitch_output);
        int input_step  = static_cast<int>(pitch_rgba);

        void* args[] = { &cu_output, &output_step, &cu_input, &input_step, &width, &height };

        CUDA_KERNEL_NODE_PARAMS_v2 params = {};
        params.func = gradationKernel.function;
        params.kernelParams = args;
        params.extra = nullptr;
        params.gridDimX = (width + 31) / 32;
        params.gridDimY = (height + 31) / 32;
        params.gridDimZ = 1;
        params.blockDimX = 32;
        params.blockDimY = 32;
        params.blockDimZ = 1;

        res = cuGraphExecKernelNodeSetParams(g_graph.graphExec, g_graph.nodeGradation, &params);
        if (res != CUDA_SUCCESS) { qWarning() << "cuGraphExecKernelNodeSetParams gradation failed" << res; return false; }
    }

    res = cuGraphLaunch(g_graph.graphExec, stream);
    if (res != CUDA_SUCCESS) { qWarning() << "cuGraphLaunch failed" << res; return false; }

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


