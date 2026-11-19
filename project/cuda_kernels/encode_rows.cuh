// ==============================================================
// RGBA（OpenGL の FBO）→ エンコード入力のタイル（NV12 / P010）を行ごとに作る共通部
//   encode_8bit.cu / encode_10bit.cu から使う。デコード側の decode_rows.cuh と対になる
//
//   1 ブロック = 1 タイルの Y 2 行と UV 1 行（kRowThreads スレッド）。1 スレッド = 横 2 画素（色差 1 組）。
//   1 スレッドがテクスチャの 2×2 を読み、Y を 4 つと、その平均から U / V を作る（色差は 2×2 の平均）。
//   タイルごとに書き方を切り替える（t.packed[q]。デコード側は読み方を切り替える）
//     普段        : NV12 / P010 の並びで書く（P010 は値を <<6）。NVENC の入力フレーム、または転送の中継へ
//     詰めたタイル: XferCodec の詰めた並び（pitch = 1 行のバイト数）へ値をそのまま書く（前処理 xfer_fwd を兼ねる）。
//                   t.delta なら行内の差分（Y は 1 つ左、UV は 2 つ左）をジグザグにして書く（xfer_fwd と同じ値）。
//                   左隣の値は、ワープ内はシャッフル、ワープ間は共有メモリ、区間の境目は繰り越しで受け取る
//   書き込みは 1 スレッドが隣り合う 2 要素（Y 2 画素、U と V）をまとめて書く（隣り合うスレッドが隣り合う位置）。
//   行の先頭が 2 要素の境界に無いとき（タイル幅が奇数の詰めた並び。例: 1366 の 2 分割 = 683）は 1 要素ずつ書く。
//   UV の行は pitch の要素数を超えて書かない（詰めた並びは XferCodec が UV を偶数要素で持つ）
//
//   Conv（色の計算。encode_8bit.cu / encode_10bit.cu が持つ）:
//     using T; static constexpr int BITS, SHIFT;      要素の型、値のビット数、NV12 / P010 の並びへ書くときのシフト
//     float3 fetch(int x, int fy)                     テクスチャの 1 画素（fy はテクスチャの行。上下反転は共通部）
//     int  luma(float3 c, int x, int y)               Y の値（x, y は画像の座標。ディザに使う）
//     void chroma(float3 avg, int x, int y, int& u, int& v)   2×2 の平均から U / V（x, y は 2×2 の左上）
//   タイルの幅が奇数でもよい（最後の画素の色差は右隣の画素と組にする。右隣は隣のタイルの画素、画像の右端では自分）
//   起動: grid((outH + 1) / 2, count)、block(kRowThreads)
// ==============================================================
#ifndef ENCODE_ROWS_CUH
#define ENCODE_ROWS_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "tiles_10bit.h"   // NV12Tiles（P010 も同じ並びで渡す）

namespace encode_rows {

constexpr int kRowThreads = 256;

// 差分（bits ビットで巡回させて符号付きに）→ ジグザグ（0, -1, 1, -2 .. → 0, 1, 2, 3 ..）。xfer_fwd と同じ
template <int BITS>
__device__ __forceinline__ int delta_code(int v, int p)
{
    int d = (v - p) & ((1 << BITS) - 1);
    if (d >= (1 << (BITS - 1))) d -= (1 << BITS);
    return int((unsigned(d) << 1) ^ unsigned(d >> 31));
}

// 隣り合う 2 要素を書く（vec: 2 要素の境界にあるのでまとめて書ける）
__device__ __forceinline__ void store2(uint8_t* p, int a, int b, bool vec)
{
    if (vec) *reinterpret_cast<uchar2*>(p) = make_uchar2(uint8_t(a), uint8_t(b));
    else     { p[0] = uint8_t(a); p[1] = uint8_t(b); }
}
__device__ __forceinline__ void store2(uint16_t* p, int a, int b, bool vec)
{
    if (vec) *reinterpret_cast<ushort2*>(p) = make_ushort2(uint16_t(a), uint16_t(b));
    else     { p[0] = uint16_t(a); p[1] = uint16_t(b); }
}

// 行の先頭が 2 要素の境界にあるか（先頭 + pitch × 行。dx は常に偶数）
template <typename T>
__device__ __forceinline__ bool aligned2(const void* p) { return (reinterpret_cast<uintptr_t>(p) & (2 * sizeof(T) - 1)) == 0; }

template <class Conv>
__device__ __forceinline__ void convert(const NV12Tiles& t, int srcW, int srcH, int flip, const Conv& conv)
{
    using T = typename Conv::T;
    const int q = blockIdx.y;
    const int r = blockIdx.x;                       // UV の行（Y はタイルの 2r と 2r+1 行）
    if (q >= t.count || 2 * r >= t.outH) return;
    const int ox = (q % t.cols) * t.outW;           // タイルの左上（画像の座標）
    const int y0 = (q / t.cols) * t.outH + 2 * r;
    if (y0 >= srcH) return;
    const int  y1   = min(y0 + 1, srcH - 1);        // 色差の 2×2 の下の行（画像の端で止める）
    const bool hasB = 2 * r + 1 < t.outH && y0 + 1 < srcH;
    const int  f0   = flip ? srcH - 1 - y0 : y0;
    const int  f1   = flip ? srcH - 1 - y1 : y1;

    T* ya = reinterpret_cast<T*>(t.y[q]  + size_t(2 * r) * t.yp[q]);
    T* yb = reinterpret_cast<T*>(t.y[q]  + size_t(2 * r + (hasB ? 1 : 0)) * t.yp[q]);
    T* uv = reinterpret_cast<T*>(t.uv[q] + size_t(r) * t.uvp[q]);
    const bool packed = t.packed[q] != 0;
    const bool delta  = packed && t.delta;
    const int  sh     = packed ? 0 : Conv::SHIFT;
    const int  pairs  = (t.outW + 1) / 2;
    const bool vya    = aligned2<T>(ya), vyb = aligned2<T>(yb), vuv = aligned2<T>(uv);
    const int  uvElems = int(t.uvp[q] / sizeof(T));   // UV の行に書ける要素数（詰めた並びは丁度、普段は pitch）

    __shared__ int edge[kRowThreads / 32][4];       // ワープの最後のスレッドの値（Y 上・Y 下の右の画素、U、V）
    __shared__ int carry[4];                        // 前の区間の最後のスレッドの値
    if (threadIdx.x < 4) carry[threadIdx.x] = 0;    // 行頭の左は 0（xfer_fwd と同じ）
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int base = 0; base < pairs; base += kRowThreads) {
        const int  i     = base + threadIdx.x;
        const int  dx    = 2 * i;                   // タイルの中の x
        const int  x0    = ox + dx;
        const bool valid = i < pairs && x0 < srcW;
        const bool has1  = valid && dx + 1 < t.outW && x0 + 1 < srcW;

        // Y（上の行 2 つ、下の行 2 つ）、U、V
        int ya0 = 0, ya1 = 0, yb0 = 0, yb1 = 0, u = 0, v = 0;
        if (valid) {
            const int    x1 = min(x0 + 1, srcW - 1);
            const float3 a = conv.fetch(x0, f0), b = conv.fetch(x1, f0);
            const float3 c = conv.fetch(x0, f1), d = conv.fetch(x1, f1);
            ya0 = conv.luma(a, x0, y0);
            if (has1) ya1 = conv.luma(b, x0 + 1, y0);
            if (hasB) {
                yb0 = conv.luma(c, x0, y0 + 1);
                if (has1) yb1 = conv.luma(d, x0 + 1, y0 + 1);
            }
            const float3 m = make_float3((a.x + b.x + c.x + d.x) * 0.25f,
                                         (a.y + b.y + c.y + d.y) * 0.25f,
                                         (a.z + b.z + c.z + d.z) * 0.25f);
            conv.chroma(m, x0, y0, u, v);
        }

        if (delta) {
            // 左隣（スレッド i-1）の右の画素の Y（上・下）と U、V を受け取る
            const int mine[4] = { ya1, yb1, u, v };
            int prev[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) prev[k] = __shfl_up_sync(0xffffffffu, mine[k], 1);
            if (lane == 31) {
                #pragma unroll
                for (int k = 0; k < 4; k++) edge[warp][k] = mine[k];
            }
            __syncthreads();
            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < 4; k++) prev[k] = warp > 0 ? edge[warp - 1][k] : carry[k];
            }
            __syncthreads();
            if (threadIdx.x == kRowThreads - 1) {   // 次の区間へ繰り越す（次の区間の 1 回目の同期の後で読む）
                #pragma unroll
                for (int k = 0; k < 4; k++) carry[k] = mine[k];
            }
            constexpr int B = Conv::BITS;
            const int za1 = delta_code<B>(ya1, ya0), zb1 = delta_code<B>(yb1, yb0);
            ya0 = delta_code<B>(ya0, prev[0]);  ya1 = za1;
            yb0 = delta_code<B>(yb0, prev[1]);  yb1 = zb1;
            u   = delta_code<B>(u, prev[2]);
            v   = delta_code<B>(v, prev[3]);
        }

        if (valid) {
            if (has1) store2(ya + dx, ya0 << sh, ya1 << sh, vya); else ya[dx] = T(ya0 << sh);
            if (hasB) {
                if (has1) store2(yb + dx, yb0 << sh, yb1 << sh, vyb); else yb[dx] = T(yb0 << sh);
            }
            if (dx + 1 < uvElems) store2(uv + dx, u << sh, v << sh, vuv);
            else                  uv[dx] = T(u << sh);
        }
    }
}

} // namespace encode_rows

#endif // ENCODE_ROWS_CUH
