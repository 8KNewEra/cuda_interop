// ==============================================================
// NVDEC のタイル（NV12 / P010）→ RGBA を行ごとに変換する共通部（decode_8bit.cu / decode_10bit.cu から使う）
//
//   1 ブロック = 1 タイルの Y 2 行と UV 1 行（kRowThreads スレッド）。
//   読み込みは 1 スレッド 16 バイト（8bit なら 16 画素、P010 なら 8 画素）× Y 2 行 ＋ UV 1 行。
//   UV の同じ 16 バイトが、同じ画素の色差（U/V 交互）になる。
//   圧縮転送の詰めたタイル（t.packed[q]）は値がそのまま（P010 も >>6 済み）で、t.delta なら行内の差分（ジグザグ）。
//     差分は行ごとに累積和（スレッド内 → ワープ → ブロック → 前の区間の繰り越し）で戻す（系列: Y 2 行、U、V）
//   戻した値は共有メモリに置き、書き込みは隣り合うスレッドが隣り合う画素を受け持つ（surface への書き込みをまとめる）。
//   条件: y / uv の先頭と pitch が 16 の倍数（VideoEngine が確かめる）。行末は 16 バイト単位で pitch の中まで読む。
//   詰めたタイルは 1 行が 16 バイトの倍数（nvgpudecode が保証）。高さが奇数なら最後のブロックは Y 1 行だけ。
//   起動: grid((srcH + 1) / 2, count)、block(kRowThreads)
// ==============================================================
#ifndef TILE_ROWS_CUH
#define TILE_ROWS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "tiles_10bit.h"   // P010Tiles（8bit の NV12 も同じ並びで渡す）

namespace tile_rows {

constexpr int kRowThreads = 256;

__device__ __forceinline__ int unzigzag(unsigned z) { return int(z >> 1) ^ -int(z & 1u); }

// T: 1 要素（uint8_t = NV12、uint16_t = P010）、BITS: 値のビット数、RAW_SHIFT: NVDEC の出力から値を取り出すシフト
// Writer: (int Y, int U, int V, int x, int y) で 1 画素を書く
template <typename T, int BITS, int RAW_SHIFT, class Writer>
__device__ __forceinline__ void convert(const P010Tiles& t, int outW, int outH, Writer write)
{
    constexpr int V    = 16 / int(sizeof(T));   // 1 スレッドの要素数
    constexpr int MASK = (1 << BITS) - 1;

    const int q = blockIdx.y;
    const int r = blockIdx.x;                    // UV の行（Y は 2r と 2r+1）
    if (q >= t.count || 2 * r >= t.srcH) return;
    const bool hasB = 2 * r + 1 < t.srcH;
    const int ox0 = (q % t.cols) * t.srcW;
    const int oy0 = (q / t.cols) * t.srcH + 2 * r;

    const uint4* ya = reinterpret_cast<const uint4*>(t.y[q]  + size_t(2 * r) * t.yp[q]);
    const uint4* yb = reinterpret_cast<const uint4*>(t.y[q]  + size_t(hasB ? 2 * r + 1 : 2 * r) * t.yp[q]);
    const uint4* uv = reinterpret_cast<const uint4*>(t.uv[q] + size_t(r) * t.uvp[q]);
    const int  vecs   = (t.srcW + V - 1) / V;
    const bool packed = t.packed[q] != 0;
    const bool delta  = packed && t.delta;

    __shared__ int warpSum[kRowThreads / 32][4];
    __shared__ int carry[4];
    __shared__ T sYa[kRowThreads * V], sYb[kRowThreads * V], sUV[kRowThreads * V];
    if (threadIdx.x < 4) carry[threadIdx.x] = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int base = 0; base < vecs; base += kRowThreads) {
        const int i = base + threadIdx.x;
        uint4 wa = make_uint4(0, 0, 0, 0), wb = wa, wc = wa;
        if (i < vecs) { wa = ya[i]; wb = yb[i]; wc = uv[i]; }
        const T* ea = reinterpret_cast<const T*>(&wa);
        const T* eb = reinterpret_cast<const T*>(&wb);
        const T* ec = reinterpret_cast<const T*>(&wc);
        int va[V], vb[V], vc[V];

        if (delta) {
            int tot[4] = { 0, 0, 0, 0 };
            #pragma unroll
            for (int j = 0; j < V; j++) {
                tot[0] += unzigzag(ea[j]); va[j] = tot[0];
                tot[1] += unzigzag(eb[j]); vb[j] = tot[1];
                tot[2 + (j & 1)] += unzigzag(ec[j]); vc[j] = tot[2 + (j & 1)];
            }
            int inc[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                inc[k] = tot[k];
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int v = __shfl_up_sync(0xffffffffu, inc[k], o);
                    if (lane >= o) inc[k] += v;
                }
            }
            if (lane == 31) {
                #pragma unroll
                for (int k = 0; k < 4; k++) warpSum[warp][k] = inc[k];
            }
            __syncthreads();
            if (warp == 0) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int s = (lane < kRowThreads / 32) ? warpSum[lane][k] : 0;
                    #pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        const int v = __shfl_up_sync(0xffffffffu, s, o);
                        if (lane >= o) s += v;
                    }
                    if (lane < kRowThreads / 32) warpSum[lane][k] = s;
                }
            }
            __syncthreads();
            int pre[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) pre[k] = (warp > 0 ? warpSum[warp - 1][k] : 0) + carry[k] + (inc[k] - tot[k]);
            #pragma unroll
            for (int j = 0; j < V; j++) {
                va[j] = (va[j] + pre[0]) & MASK;
                vb[j] = (vb[j] + pre[1]) & MASK;
                vc[j] = (vc[j] + pre[2 + (j & 1)]) & MASK;
            }
            __syncthreads();
            if (threadIdx.x < 4) carry[threadIdx.x] += warpSum[kRowThreads / 32 - 1][threadIdx.x];
            __syncthreads();
        } else {
            const int sh = packed ? 0 : RAW_SHIFT;
            #pragma unroll
            for (int j = 0; j < V; j++) { va[j] = ea[j] >> sh; vb[j] = eb[j] >> sh; vc[j] = ec[j] >> sh; }
        }

        #pragma unroll
        for (int j = 0; j < V; j++) {
            sYa[threadIdx.x * V + j] = T(va[j]);
            sYb[threadIdx.x * V + j] = T(vb[j]);
            sUV[threadIdx.x * V + j] = T(vc[j]);
        }
        __syncthreads();
        // 書き込み: 区間の画素 p（= x - V*base）を隣り合うスレッドで
        const int px = min(kRowThreads * V, t.srcW - V * base);
        for (int p = threadIdx.x; p < px; p += kRowThreads) {
            const int ox = ox0 + V * base + p;
            if (ox >= outW) break;
            const int u = sUV[p & ~1], v = sUV[(p & ~1) + 1];
            if (oy0 < outH)             write(int(sYa[p]), u, v, ox, oy0);
            if (hasB && oy0 + 1 < outH) write(int(sYb[p]), u, v, ox, oy0 + 1);
        }
        __syncthreads();   // 次の区間で共有メモリを上書きする前に
    }
}

} // namespace tile_rows

#endif // TILE_ROWS_CUH
