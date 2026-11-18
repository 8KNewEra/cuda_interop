// ==============================================================
// 8bit 経路: RGBA8 の FBO からヒストグラム（R / G / B 各 256 段）
//   calc_histogram_shared_kernel : FBO（texture object）から集計（ブロック内は shared メモリ）
//   histgram_status_kernel       : 最小・最大・平均と縦軸の最大
//   histgram_normalize_kernel    : 描画用の頂点（VBO）へ正規化して書く
//   注意: HistData / HistStats は __global__.h の同名の構造体と同じ並びにすること
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
// ==============================================================
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

struct HistStats {
    int min_r, max_r;
    int min_g, max_g;
    int min_b, max_b;
    double avg_r, avg_g, avg_b;
    int max_y_axis;
};

extern "C"
__global__ void calc_histogram_shared_kernel(
    HistData* Histdata,
    cudaTextureObject_t texObj,
    int width, int height)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ unsigned int block_hist_r[256];
    __shared__ unsigned int block_hist_g[256];
    __shared__ unsigned int block_hist_b[256];

    __shared__ unsigned int local_max_r[256];
    __shared__ unsigned int local_max_g[256];
    __shared__ unsigned int local_max_b[256];

    if (tid < 256) {
        block_hist_r[tid] = 0;
        block_hist_g[tid] = 0;
        block_hist_b[tid] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x < width && y < height) {
        uchar4 p = tex2D<uchar4>(texObj, x + 0.5f, y + 0.5f);
        atomicAdd(&block_hist_r[p.x], 1u);
        atomicAdd(&block_hist_g[p.y], 1u);
        atomicAdd(&block_hist_b[p.z], 1u);
    }
    __syncthreads();

    if (tid < 256) {
        unsigned int r = block_hist_r[tid];
        unsigned int g = block_hist_g[tid];
        unsigned int b = block_hist_b[tid];

        if (r) atomicAdd(&Histdata->hist_r[tid], r);
        if (g) atomicAdd(&Histdata->hist_g[tid], g);
        if (b) atomicAdd(&Histdata->hist_b[tid], b);

        local_max_r[tid] = Histdata->hist_r[tid];
        local_max_g[tid] = Histdata->hist_g[tid];
        local_max_b[tid] = Histdata->hist_b[tid];
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_max_r[tid] = max(local_max_r[tid], local_max_r[tid + stride]);
            local_max_g[tid] = max(local_max_g[tid], local_max_g[tid + stride]);
            local_max_b[tid] = max(local_max_b[tid], local_max_b[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(&Histdata->max_r, local_max_r[0]);
        atomicMax(&Histdata->max_g, local_max_g[0]);
        atomicMax(&Histdata->max_b, local_max_b[0]);
    }
}

extern "C"
__global__
void histgram_status_kernel(
    const HistData* Histdata,
    HistStats* out_stats)
{
    int x = threadIdx.x;

    __shared__ int s_min_r[256], s_max_r[256];
    __shared__ int s_min_g[256], s_max_g[256];
    __shared__ int s_min_b[256], s_max_b[256];

    __shared__ unsigned long long s_sum_r[256], s_count_r[256];
    __shared__ unsigned long long s_sum_g[256], s_count_g[256];
    __shared__ unsigned long long s_sum_b[256], s_count_b[256];

    unsigned int r = Histdata->hist_r[x];
    unsigned int g = Histdata->hist_g[x];
    unsigned int b = Histdata->hist_b[x];

    s_min_r[x] = (r > 0) ? x : 9999;
    s_min_g[x] = (g > 0) ? x : 9999;
    s_min_b[x] = (b > 0) ? x : 9999;

    s_max_r[x] = (r > 0) ? x : -1;
    s_max_g[x] = (g > 0) ? x : -1;
    s_max_b[x] = (b > 0) ? x : -1;

    s_sum_r[x]   = (unsigned long long)x * r;
    s_sum_g[x]   = (unsigned long long)x * g;
    s_sum_b[x]   = (unsigned long long)x * b;

    s_count_r[x] = r;
    s_count_g[x] = g;
    s_count_b[x] = b;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (x < stride) {
            s_min_r[x] = min(s_min_r[x], s_min_r[x + stride]);
            s_min_g[x] = min(s_min_g[x], s_min_g[x + stride]);
            s_min_b[x] = min(s_min_b[x], s_min_b[x + stride]);

            s_max_r[x] = max(s_max_r[x], s_max_r[x + stride]);
            s_max_g[x] = max(s_max_g[x], s_max_g[x + stride]);
            s_max_b[x] = max(s_max_b[x], s_max_b[x + stride]);

            s_sum_r[x] += s_sum_r[x + stride];
            s_sum_g[x] += s_sum_g[x + stride];
            s_sum_b[x] += s_sum_b[x + stride];

            s_count_r[x] += s_count_r[x + stride];
            s_count_g[x] += s_count_g[x + stride];
            s_count_b[x] += s_count_b[x + stride];
        }
        __syncthreads();
    }

    if (x == 0) {
        HistStats out;

        out.min_r = (s_min_r[0] == 9999) ? -1 : s_min_r[0];
        out.min_g = (s_min_g[0] == 9999) ? -1 : s_min_g[0];
        out.min_b = (s_min_b[0] == 9999) ? -1 : s_min_b[0];

        out.max_r = s_max_r[0];
        out.max_g = s_max_g[0];
        out.max_b = s_max_b[0];

        out.avg_r = (s_count_r[0] > 0)
            ? double(s_sum_r[0]) / double(s_count_r[0]) : 0.0;

        out.avg_g = (s_count_g[0] > 0)
            ? double(s_sum_g[0]) / double(s_count_g[0]) : 0.0;

        out.avg_b = (s_count_b[0] > 0)
            ? double(s_sum_b[0]) / double(s_count_b[0]) : 0.0;

        out.avg_r = round(out.avg_r * 10.0) / 10.0;
        out.avg_g = round(out.avg_g * 10.0) / 10.0;
        out.avg_b = round(out.avg_b * 10.0) / 10.0;

        out.max_y_axis = max(Histdata->max_r,max(Histdata->max_g, Histdata->max_b));

        *out_stats = out;
    }
}

extern "C"
__global__ void histgram_normalize_kernel(
    float* vbo,
    int num_bins,
    HistData* Histdata,
    HistStats* input_stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= num_bins) return;

    float inv = 1.0f / max(1u, input_stats->max_y_axis);

    int r_offset = 0;
    int g_offset = num_bins * 3;
    int b_offset = num_bins * 3 * 2;

    float i = (float)x / (num_bins - 1);

    vbo[r_offset + x*3 + 0] = i;
    vbo[r_offset + x*3 + 1] = Histdata->hist_r[x] * inv;
    vbo[r_offset + x*3 + 2] = 0.0f;

    vbo[g_offset + x*3 + 0] = i;
    vbo[g_offset + x*3 + 1] = Histdata->hist_g[x] * inv;
    vbo[g_offset + x*3 + 2] = 0.0f;

    vbo[b_offset + x*3 + 0] = i;
    vbo[b_offset + x*3 + 1] = Histdata->hist_b[x] * inv;
    vbo[b_offset + x*3 + 2] = 0.0f;
}
