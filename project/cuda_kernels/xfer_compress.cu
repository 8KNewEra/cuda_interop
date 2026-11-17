// ==============================================================
// マルチ GPU デコードの圧縮転送（src/videoprocess/xfer_codec.*）で使うカーネル
//
//   流れ（1 フレーム・別 GPU のタイルごと）
//     デコード側 GPU : xfer_fwd（NV12 / P010 → 詰めた並び。差分・ジグザグ）→ nvCOMP で圧縮
//                     → xfer_pack（圧縮チャンクを詰めて、メインメモリ（ページ固定・マップ済み）へ直接書く。
//                       圧縮後の実際のバイト数だけが PCIe を流れる）
//     表示用 GPU     : xfer_fetch（メインメモリ → GPU。チャンクの位置と大きさも作る）→ nvCOMP で展開
//                     → xfer_inv（差分を戻して NV12 / P010 の転送先へ）
//   CPU は待たない（圧縮後の大きさはメインメモリ上のヘッダで受け渡す）
//
//   差分（delta）: 行の中で LAG 個左の値との差（Y は 1、UV は U/V が交互なので 2）。差はジグザグ
//     （0,-1,1,-2.. → 0,1,2,3..）にして小さい正の数にする。滑らかな画ほど 0 付近に集まり縮みやすい
//   P010 は値が上位 10bit に入っているので、>>6 してから扱う（戻すときに <<6。下位 6bit は常に 0）
//   起動口はこのファイルの extern "C" のホスト関数（nvgpudecode / xfer_codec.cpp から呼ぶ）
// ==============================================================
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t zigzag(int32_t d)        { return (uint32_t(d) << 1) ^ uint32_t(d >> 31); }
__device__ __forceinline__ int32_t  unzigzag(uint32_t z)     { return int32_t(z >> 1) ^ -int32_t(z & 1u); }

template <typename T>
__device__ __forceinline__ uint32_t load_val(const uint8_t* row, int i, int shift)
{
    return uint32_t(reinterpret_cast<const T*>(row)[i]) >> shift;
}

// 順変換: pitched → 詰めた並び（1 スレッド 1 要素）
//   bps: 1 要素のバイト数（1 = NV12、2 = P010）、lag: 差を取る相手（1 = Y、2 = UV）、bits: 値のビット数（8 / 10）
template <typename T>
__global__ void xfer_fwd_kernel(const uint8_t* src, size_t srcPitch, uint8_t* dst,
                                int elems, int rows, int lag, int shift, int bits, int delta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    if (x >= elems || y >= rows) return;
    const uint8_t* row = src + size_t(y) * srcPitch;
    uint32_t v = load_val<T>(row, x, shift);
    if (delta) {
        const uint32_t p = (x >= lag) ? load_val<T>(row, x - lag, shift) : 0u;
        // 差は bits ビットの範囲で巡回させてから符号付きに（-2^(bits-1) .. 2^(bits-1)-1）
        const uint32_t mask = (1u << bits) - 1u;
        int32_t d = int32_t((v - p) & mask);
        if (d >= (1 << (bits - 1))) d -= (1 << bits);
        v = zigzag(d);   // 0 .. 2^bits - 1
    }
    reinterpret_cast<T*>(dst + size_t(y) * size_t(elems) * sizeof(T))[x] = T(v);
}

// 逆変換: 詰めた並び → pitched。差分なら行ごとに累積和（1 ブロック 1 行、LAG 系列ごとに走査）
constexpr int kScanThreads = 256;

template <typename T, int LAG>
__global__ void xfer_inv_kernel(const uint8_t* src, uint8_t* dst, size_t dstPitch,
                                int elems, int rows, int shift, int bits, int delta)
{
    const int y = blockIdx.x;
    if (y >= rows) return;
    const T* in  = reinterpret_cast<const T*>(src + size_t(y) * size_t(elems) * sizeof(T));
    T*       out = reinterpret_cast<T*>(dst + size_t(y) * dstPitch);
    const uint32_t mask = (1u << bits) - 1u;

    if (!delta) {
        for (int i = threadIdx.x; i < elems; i += blockDim.x) out[i] = T(uint32_t(in[i]) << shift);
        return;
    }

    // LAG 個の系列（UV なら U と V）をまとめて 1 グループとして走査する
    __shared__ int32_t warpSum[kScanThreads / 32][LAG];
    __shared__ int32_t carry[LAG];
    if (threadIdx.x < LAG) carry[threadIdx.x] = 0;
    __syncthreads();

    const int groups = elems / LAG;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int base = 0; base < groups; base += kScanThreads) {
        const int g = base + threadIdx.x;
        int32_t v[LAG];
        #pragma unroll
        for (int k = 0; k < LAG; k++) v[k] = (g < groups) ? unzigzag(uint32_t(in[g * LAG + k])) : 0;
        // ワープ内の累積和
        #pragma unroll
        for (int k = 0; k < LAG; k++) {
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int32_t t = __shfl_up_sync(0xffffffffu, v[k], o);
                if (lane >= o) v[k] += t;
            }
        }
        if (lane == 31) {
            #pragma unroll
            for (int k = 0; k < LAG; k++) warpSum[warp][k] = v[k];
        }
        __syncthreads();
        // ワープ間（先頭ワープが合計を走査）
        if (warp == 0) {
            #pragma unroll
            for (int k = 0; k < LAG; k++) {
                int32_t s = (lane < kScanThreads / 32) ? warpSum[lane][k] : 0;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int32_t t = __shfl_up_sync(0xffffffffu, s, o);
                    if (lane >= o) s += t;
                }
                if (lane < kScanThreads / 32) warpSum[lane][k] = s;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < LAG; k++) {
            const int32_t prefix = (warp > 0 ? warpSum[warp - 1][k] : 0) + carry[k];
            if (g < groups) out[g * LAG + k] = T((uint32_t(v[k] + prefix) & mask) << shift);
        }
        __syncthreads();
        if (threadIdx.x < LAG) carry[threadIdx.x] += warpSum[kScanThreads / 32 - 1][threadIdx.x];
        __syncthreads();
    }
}

// 圧縮チャンクの位置（16 バイト境界）を決めてヘッダへ書く（1 スレッド。チャンクは数千まで）
//   hdr: [0] = チャンク数、[1] = 合計バイト、[2..2+n) = 大きさ、[2+n..2+2n) = 位置（データ域の先頭から）
__global__ void xfer_scan_kernel(const size_t* compBytes, int n, uint64_t* hdr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t off = 0;
    for (int i = 0; i < n; i++) {
        const uint64_t s = compBytes[i];
        hdr[2 + i]     = s;
        hdr[2 + n + i] = off;
        off += (s + 15) & ~uint64_t(15);
    }
    hdr[0] = uint64_t(n);
    hdr[1] = off;
}

// 圧縮チャンクをデータ域へ詰める（1 ブロック 1 チャンク、16 バイト単位。読む側も 16 バイト単位で余分を読む）
//   ブロック数は少なめ（kPackBlocks）にしてチャンクを順に回る。書き出しは PCIe の速さで決まるので、
//   SM を埋めないようにして、次のフレームの圧縮（別ストリーム）と並んで動けるようにする
__global__ void xfer_pack_kernel(const void* const* slots, const uint64_t* hdr, int n, uint8_t* data)
{
    for (int b = blockIdx.x; b < n; b += gridDim.x) {
        const uint64_t bytes = hdr[2 + b], off = hdr[2 + n + b];
        const uint4* s = reinterpret_cast<const uint4*>(slots[b]);
        uint4*       d = reinterpret_cast<uint4*>(data + off);
        const uint64_t words = (bytes + 15) / 16;
        for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) d[i] = s[i];
    }
}

// メインメモリ → 表示用 GPU の残り。nvCOMP の展開に渡すチャンクの位置と大きさも作る
//   データの先頭 copied バイトは DMA（cudaMemcpyAsync。コピーエンジンなので SM を使わない）で済んでいる。
//   見込みを超えた分（copied より後ろ）だけ、カーネルがメインメモリから直接読む（めったに無い）
//   hdr は表示用 GPU に DMA で写したヘッダ
__global__ void xfer_fetch_kernel(const uint64_t* hdr, int n, const uint8_t* hostData, uint8_t* devData,
                                  uint64_t copied, const void** compPtrs, size_t* compBytes)
{
    const int b = blockIdx.x;
    if (b >= n) return;
    const uint64_t bytes = hdr[2 + b], off = hdr[2 + n + b];
    const uint64_t end = off + ((bytes + 15) & ~uint64_t(15));
    if (end > copied) {
        const uint64_t from = off > copied ? off : copied;   // copied は 16 の倍数
        const uint4* s = reinterpret_cast<const uint4*>(hostData + from);
        uint4*       d = reinterpret_cast<uint4*>(devData + from);
        const uint64_t words = (end - from) / 16;
        for (uint64_t i = threadIdx.x; i < words; i += blockDim.x) d[i] = s[i];
    }
    if (threadIdx.x == 0) {
        compPtrs[b]  = devData + off;
        compBytes[b] = size_t(bytes);
    }
}

} // namespace

// ==============================================================
// ホスト側の起動口（呼ぶ前に対象 GPU を cudaSetDevice しておく）
// ==============================================================
extern "C" cudaError_t xfer_fwd(const uint8_t* src, size_t srcPitch, uint8_t* dst, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st)
{
    const int shift = (bps == 2) ? 6 : 0;
    const int bits  = (bps == 2) ? 10 : 8;
    dim3 block(256), grid((elems + 255) / 256, rows);
    if (bps == 2) xfer_fwd_kernel<uint16_t><<<grid, block, 0, st>>>(src, srcPitch, dst, elems, rows, lag, shift, bits, delta);
    else          xfer_fwd_kernel<uint8_t ><<<grid, block, 0, st>>>(src, srcPitch, dst, elems, rows, lag, shift, bits, delta);
    return cudaGetLastError();
}

extern "C" cudaError_t xfer_inv(const uint8_t* src, uint8_t* dst, size_t dstPitch, int elems, int rows,
                                int bps, int lag, int delta, cudaStream_t st)
{
    const int shift = (bps == 2) ? 6 : 0;
    const int bits  = (bps == 2) ? 10 : 8;
    dim3 block(kScanThreads), grid(rows);
    if (bps == 2) {
        if (lag == 2) xfer_inv_kernel<uint16_t, 2><<<grid, block, 0, st>>>(src, dst, dstPitch, elems, rows, shift, bits, delta);
        else          xfer_inv_kernel<uint16_t, 1><<<grid, block, 0, st>>>(src, dst, dstPitch, elems, rows, shift, bits, delta);
    } else {
        if (lag == 2) xfer_inv_kernel<uint8_t, 2><<<grid, block, 0, st>>>(src, dst, dstPitch, elems, rows, shift, bits, delta);
        else          xfer_inv_kernel<uint8_t, 1><<<grid, block, 0, st>>>(src, dst, dstPitch, elems, rows, shift, bits, delta);
    }
    return cudaGetLastError();
}

// devHdr: デコード側 GPU のヘッダ（位置を決める作業用）、hostHdr / hostData: メインメモリ（ページ固定・マップ済み）
//   データはカーネルがメインメモリへ直接書く（圧縮後の大きさを CPU が知らなくても、実際のバイト数だけ PCIe を流れる）
//   ヘッダは小さいので DMA でコピーする
extern "C" cudaError_t xfer_pack(const void* const* slots, const size_t* compBytes, int n,
                                 uint64_t* devHdr, uint64_t* hostHdr, uint8_t* hostData, cudaStream_t st)
{
    xfer_scan_kernel<<<1, 1, 0, st>>>(compBytes, n, devHdr);
    constexpr int kPackBlocks = 32;
    xfer_pack_kernel<<<n < kPackBlocks ? n : kPackBlocks, 256, 0, st>>>(slots, devHdr, n, hostData);
    cudaMemcpyAsync(hostHdr, devHdr, sizeof(uint64_t) * (2 + 2 * size_t(n)), cudaMemcpyDeviceToHost, st);
    return cudaGetLastError();
}

// hostHdr / hostData: メインメモリ（ページ固定）、devHdr / devData: 表示用 GPU
//   ヘッダと、データの先頭 copyBytes バイト（圧縮後の大きさの見込み）は DMA で写す。
//   CPU は今のフレームの圧縮後の大きさを知らない（待たない）ので、見込みを超えた分はカーネルが拾う
extern "C" cudaError_t xfer_fetch(const uint64_t* hostHdr, uint64_t* devHdr, int n, const uint8_t* hostData,
                                  uint8_t* devData, size_t copyBytes, const void** compPtrs, size_t* compBytes,
                                  cudaStream_t st)
{
    copyBytes &= ~size_t(15);
    cudaMemcpyAsync(devHdr, hostHdr, sizeof(uint64_t) * (2 + 2 * size_t(n)), cudaMemcpyHostToDevice, st);
    if (copyBytes) cudaMemcpyAsync(devData, hostData, copyBytes, cudaMemcpyHostToDevice, st);
    xfer_fetch_kernel<<<n, 256, 0, st>>>(devHdr, n, hostData, devData, uint64_t(copyBytes), compPtrs, compBytes);
    return cudaGetLastError();
}
