#ifndef DECODE_GPU_MAP_H
#define DECODE_GPU_MAP_H

// マルチ GPU デコードのタイル割り振り（詳細は docs/MULTI_GPU_DECODE.md）
//   GPU ごとの重み（GPUInfo::decode_weight）で最大剰余法により按分する。戻り値はタイルごとの CUDA デバイス番号で、
//   表示用 GPU のタイルが先頭。使うのは nvgpudecode（開くとき）と DecodeSettings（設定画面の見込み）

#include "src/main/__global__.h"
#include <algorithm>
#include <cstring>
#include <vector>

// 重み（未設定 = -1 は表示用 GPU だけ 10、ほかは 0）
inline int effectiveDecodeWeight(const GPUInfo& g)
{
    if (g.decode_weight >= 0) return std::clamp(g.decode_weight, 0, 10);
    return g.deviceID == g_openglDeviceID ? 10 : 0;
}

// この GPU の NVDEC でデコードできるか（codec は avcodec_get_name の名前）
inline bool gpuCanDecode(const GPUInfo& g, const char* codec)
{
    if (codec && std::strcmp(codec, "av1") == 0)
        return g.CC_major > 8 || (g.CC_major == 8 && g.CC_minor >= 6);   // AV1 の NVDEC は Ampere（GA10x）以降
    return true;   // H.264 / HEVC は対象 GPU（sm_75 以降）すべて
}

// gpus: 重みを読む GPU の一覧（設定画面は編集中の値を渡す）。使える GPU が無ければ全部を表示用 GPU に
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

    // 端数は 余りの大きい順 → 重みの大きい順 → 表示用 GPU → 番号の小さい順
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

    for (size_t i = 0; i < c.size(); ++i) if (c[i].primary)  out.insert(out.end(), cnt[i], c[i].id);
    for (size_t i = 0; i < c.size(); ++i) if (!c[i].primary) out.insert(out.end(), cnt[i], c[i].id);
    return out;
}

#endif // DECODE_GPU_MAP_H
