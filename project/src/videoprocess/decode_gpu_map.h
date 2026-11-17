#ifndef DECODE_GPU_MAP_H
#define DECODE_GPU_MAP_H

// ==============================================================
// マルチ GPU デコード: タイルをどの GPU の NVDEC でデコードするか
//
//   GPU ごとの「デコードの重み」（GPUInfo::decode_weight、0..10）でタイルを按分する。
//     ・未設定（-1）は、表示用 GPU（g_openglDeviceID）= 10、それ以外 = 0 とみなす（= 従来どおり 1 枚）
//     ・そのコーデックをデコードできない GPU は外す（AV1 は Ampere 世代 sm_86 以降）
//     ・按分は最大剰余法。端数は重みの大きい GPU → 表示用 GPU → 番号の小さい順に配る
//     ・どの GPU も使えなければ全部を表示用 GPU に
//   戻り値はタイルごとの CUDA デバイス番号（表示用 GPU のタイルを先に並べる）
//
//   表示用 GPU 以外でデコードしたタイルは、nvgpudecode::publish_decode_frame が表示用 GPU へ転送する
//   （cudaMemcpy3DPeerAsync。GPU 同士で直接送れない環境ではドライバがメインメモリを経由する）
//   使うのは nvgpudecode（ファイルを開くとき）と DecodeSettings（設定画面の見込み表示）
// ==============================================================

#include "src/main/__global__.h"
#include <algorithm>
#include <cstring>
#include <vector>

// 表示に使う重み（未設定なら表示用 GPU = 10、ほか = 0）
inline int effectiveDecodeWeight(const GPUInfo& g)
{
    if (g.decode_weight >= 0) return std::clamp(g.decode_weight, 0, 10);
    return g.deviceID == g_openglDeviceID ? 10 : 0;
}

// この GPU の NVDEC でデコードできるか（codec は avcodec_get_name の名前: "h264" / "hevc" / "av1"）
inline bool gpuCanDecode(const GPUInfo& g, const char* codec)
{
    if (codec && std::strcmp(codec, "av1") == 0)
        return g.CC_major > 8 || (g.CC_major == 8 && g.CC_minor >= 6);   // AV1 の NVDEC は Ampere（GA10x）以降
    return true;   // H.264 / HEVC（4:2:0 8bit / 10bit）は対象 GPU（sm_75 以降）すべて
}

// nTiles 本のタイルを GPU に割り振る（タイルごとのデバイス番号）
//   gpus: 重みを読む GPU の一覧（既定は g_GPUInfo。設定画面は編集中の値で見込みを出す）
inline std::vector<int> assignDecodeTiles(int nTiles, const char* codec,
                                          const std::vector<GPUInfo>& gpus = g_GPUInfo)
{
    struct Cand { int id; int w; bool primary; };
    std::vector<Cand> c;
    int sum = 0;
    for (const GPUInfo& g : gpus) {
        const int w = effectiveDecodeWeight(g);
        if (w <= 0 || !gpuCanDecode(g, codec)) continue;
        c.push_back({ g.deviceID, w, g.deviceID == g_openglDeviceID });
        sum += w;
    }
    std::vector<int> out;
    if (nTiles <= 0) return out;
    if (c.empty() || sum <= 0) return std::vector<int>(nTiles, g_openglDeviceID);

    // 最大剰余法
    std::vector<int> cnt(c.size(), 0);
    std::vector<double> rem(c.size(), 0.0);
    int given = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        const double q = double(nTiles) * c[i].w / sum;
        cnt[i] = int(q);
        rem[i] = q - cnt[i];
        given += cnt[i];
    }
    std::vector<size_t> order(c.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (rem[a] != rem[b]) return rem[a] > rem[b];
        if (c[a].w != c[b].w) return c[a].w > c[b].w;
        if (c[a].primary != c[b].primary) return c[a].primary;
        return c[a].id < c[b].id;
    });
    for (size_t k = 0; given < nTiles; k = (k + 1) % order.size()) { ++cnt[order[k]]; ++given; }

    // 表示用 GPU のタイルを先に（転送しないタイルが先頭に来る）
    for (size_t i = 0; i < c.size(); ++i) if (c[i].primary)  out.insert(out.end(), cnt[i], c[i].id);
    for (size_t i = 0; i < c.size(); ++i) if (!c[i].primary) out.insert(out.end(), cnt[i], c[i].id);
    return out;
}

#endif // DECODE_GPU_MAP_H
