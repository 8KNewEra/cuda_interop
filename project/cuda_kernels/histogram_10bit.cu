// ==============================================================
// 10bit 経路: RGBA16 の FBO からヒストグラム（R / G / B 各 1024 段 = 10bit の値ごと）
//   calc_histogram_rgba16_kernel : FBO（texture object）から集計（ブロック内は shared メモリ）
//   histgram10_status_kernel     : 最小・最大・平均（0..1023）と縦軸の最大
//   histgram10_normalize_kernel  : 描画用の頂点（VBO、1024 点 × 3 色）へ正規化して書く
//   8bit 経路（histogram_8bit.cu、256 段）と仕組みは同じ。段数が違うので集計先は HistData10。
//   統計は 8bit 経路と同じ HistStats に入れる（__global__.h と同じ並びにすること）
//   起動口は CUDA_ImageProcess（src/imageprocess/cuda_imageprocess.cpp）
// ==============================================================
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include "histogram_10bit.h"   // HistData10 / HIST10_BINS

struct HistStats {
    int min_r, max_r;
    int min_g, max_g;
    int min_b, max_b;
    double avg_r, avg_g, avg_b;
    int max_y_axis;
};

// 16bit（0..65535）→ 10bit（0..1023）。四捨五入
__device__ __forceinline__ unsigned int to10(unsigned int v)
{
    return (v * 1023u + 32767u) / 65535u;
}

// 集計。ブロック（16×16）ごとに shared メモリへ数えてから全体へ足す
extern "C"
__global__ void calc_histogram_rgba16_kernel(
    HistData10* Histdata,
    cudaTextureObject_t texObj,
    int width, int height)
{
    const int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    __shared__ unsigned int sr[HIST10_BINS];
    __shared__ unsigned int sg[HIST10_BINS];
    __shared__ unsigned int sb[HIST10_BINS];
    for (int i = tid; i < HIST10_BINS; i += threads) { sr[i] = 0; sg[i] = 0; sb[i] = 0; }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // 16bit → 10bit（0..1023 に四捨五入。スポイト（VideoEngine::pickCollect）と同じ式）
        ushort4 p = tex2D<ushort4>(texObj, x + 0.5f, y + 0.5f);
        atomicAdd(&sr[to10(p.x)], 1u);
        atomicAdd(&sg[to10(p.y)], 1u);
        atomicAdd(&sb[to10(p.z)], 1u);
    }
    __syncthreads();

    for (int i = tid; i < HIST10_BINS; i += threads) {
        if (sr[i]) atomicAdd(&Histdata->hist_r[i], sr[i]);
        if (sg[i]) atomicAdd(&Histdata->hist_g[i], sg[i]);
        if (sb[i]) atomicAdd(&Histdata->hist_b[i], sb[i]);
    }
}

// 統計（1 ブロック 256 スレッド。1 スレッドが 4 段ずつ受け持つ）
extern "C"
__global__ void histgram10_status_kernel(
    const HistData10* Histdata,
    HistStats* out_stats)
{
    const int t = threadIdx.x;   // 0..255

    __shared__ int s_min[3][256], s_max[3][256];
    __shared__ unsigned int s_peak[3][256];
    __shared__ unsigned long long s_sum[3][256], s_cnt[3][256];

    const unsigned int* h[3] = { Histdata->hist_r, Histdata->hist_g, Histdata->hist_b };
    for (int c = 0; c < 3; c++) {
        int mn = 1 << 30, mx = -1;
        unsigned int peak = 0;
        unsigned long long sum = 0, cnt = 0;
        for (int k = 0; k < HIST10_BINS / 256; k++) {
            const int b = t * (HIST10_BINS / 256) + k;
            const unsigned int v = h[c][b];
            if (v) { mn = min(mn, b); mx = max(mx, b); }
            peak = max(peak, v);
            sum += (unsigned long long)b * v;
            cnt += v;
        }
        s_min[c][t] = mn; s_max[c][t] = mx; s_peak[c][t] = peak;
        s_sum[c][t] = sum; s_cnt[c][t] = cnt;
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            for (int c = 0; c < 3; c++) {
                s_min[c][t]  = min(s_min[c][t], s_min[c][t + stride]);
                s_max[c][t]  = max(s_max[c][t], s_max[c][t + stride]);
                s_peak[c][t] = max(s_peak[c][t], s_peak[c][t + stride]);
                s_sum[c][t] += s_sum[c][t + stride];
                s_cnt[c][t] += s_cnt[c][t + stride];
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        HistStats out;
        double avg[3];
        for (int c = 0; c < 3; c++) {
            avg[c] = s_cnt[c][0] ? double(s_sum[c][0]) / double(s_cnt[c][0]) : 0.0;
            avg[c] = round(avg[c] * 10.0) / 10.0;
        }
        out.min_r = (s_min[0][0] == (1 << 30)) ? -1 : s_min[0][0];
        out.min_g = (s_min[1][0] == (1 << 30)) ? -1 : s_min[1][0];
        out.min_b = (s_min[2][0] == (1 << 30)) ? -1 : s_min[2][0];
        out.max_r = s_max[0][0];
        out.max_g = s_max[1][0];
        out.max_b = s_max[2][0];
        out.avg_r = avg[0]; out.avg_g = avg[1]; out.avg_b = avg[2];
        out.max_y_axis = (int)max(s_peak[0][0], max(s_peak[1][0], s_peak[2][0]));
        *out_stats = out;
    }
}

// 描画用の頂点（x = 0..1、y = 度数 / 縦軸の最大）。並びは R → G → B の順に num_bins 点ずつ
extern "C"
__global__ void histgram10_normalize_kernel(
    float* vbo,
    const HistData10* Histdata,
    const HistStats* input_stats)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= HIST10_BINS) return;

    const float inv = 1.0f / max(1, input_stats->max_y_axis);
    const float i   = (float)x / (HIST10_BINS - 1);
    const int g_off = HIST10_BINS * 3;
    const int b_off = HIST10_BINS * 3 * 2;

    vbo[x * 3 + 0] = i;  vbo[x * 3 + 1] = Histdata->hist_r[x] * inv;  vbo[x * 3 + 2] = 0.0f;
    vbo[g_off + x * 3 + 0] = i;  vbo[g_off + x * 3 + 1] = Histdata->hist_g[x] * inv;  vbo[g_off + x * 3 + 2] = 0.0f;
    vbo[b_off + x * 3 + 0] = i;  vbo[b_off + x * 3 + 1] = Histdata->hist_b[x] * inv;  vbo[b_off + x * 3 + 2] = 0.0f;
}
