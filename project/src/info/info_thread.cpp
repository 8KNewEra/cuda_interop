#include "src/info/info_thread.h"
#include "qtimer.h"
#include <nvml.h>
#include <QDebug>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <utility>
#include <cmath>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <cwchar>
#include <vector>

// GPU の 3D / CUDA エンジン使用率（Windows のパフォーマンスカウンタ。タスクマネージャーと同じ値）
//   NVML では 3D・CUDA を分けて取れないので PDH の "GPU Engine" を読む。
//   インスタンス名は "pid_<n>_luid_0x<High>_0x<Low>_phys_<n>_eng_<n>_engtype_<種類>"。
//   種類 "3D" → 3D、"Cuda" / "Compute_0" など（ドライバ・設定で名前が変わる）→ CUDA。
//   エンジンごとに全プロセスの値を足し、同じアダプタ（LUID）・同じ種類のエンジンの最大を使う。
//   アダプタは CUDA の deviceProp.luid で突き合わせる（TCC モードなど LUID が無い GPU は -1）。
//   注意: CUDA と OpenGL の連携（マップ・コピー）の負荷は 3D 側に乗ることがある。
namespace {
using Luid = std::pair<unsigned, unsigned>;   // {High, Low}

struct EngineUsage { int g3d = -1, cuda = -1; };   // -1 = そのエンジンのデータなし

class GpuEngineCounter {
public:
    ~GpuEngineCounter() { if (query_) PdhCloseQuery(query_); }

    // 最後の sample() で値が取れたか（false なら全 GPU を「取れない」扱いにする）
    bool valid() const { return valid_; }

    // LUID → 3D / CUDA の使用率（%）
    std::map<Luid, EngineUsage> sample()
    {
        std::map<Luid, EngineUsage> out;
        valid_ = false;
        if (!open()) return out;
        if (PdhCollectQueryData(query_) != ERROR_SUCCESS) return out;
        DWORD size = 0, count = 0;
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &size, &count, nullptr) != (PDH_STATUS)PDH_MORE_DATA)
            return out;
        std::vector<unsigned char> buf(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_DOUBLE, &size, &count, items) != ERROR_SUCCESS)
            return out;

        // {LUID, 種類（0 = 3D、1 = CUDA）, エンジン番号} → 全プロセスの合計
        std::map<std::tuple<Luid, int, int>, double> perEngine;
        for (DWORD i = 0; i < count; ++i) {
            const wchar_t* name = items[i].szName;
            const wchar_t* t = name ? std::wcsstr(name, L"engtype_") : nullptr;
            if (!t) continue;
            t += 8;
            int kind;
            if (std::wcscmp(t, L"3D") == 0)                                         kind = 0;
            else if (std::wcsncmp(t, L"Cuda", 4) == 0 || std::wcsncmp(t, L"Compute", 7) == 0) kind = 1;
            else continue;
            const DWORD st = items[i].FmtValue.CStatus;
            if (st != PDH_CSTATUS_VALID_DATA && st != PDH_CSTATUS_NEW_DATA) continue;
            const wchar_t* p = std::wcsstr(name, L"luid_0x");
            unsigned hi = 0, lo = 0;
            if (!p || std::swscanf(p, L"luid_0x%x_0x%x", &hi, &lo) != 2) continue;
            const wchar_t* e = std::wcsstr(name, L"_eng_");
            const int eng = e ? _wtoi(e + 5) : 0;
            perEngine[{Luid{hi, lo}, kind, eng}] += items[i].FmtValue.doubleValue;
        }
        for (const auto& [key, v] : perEngine) {
            EngineUsage& u = out[std::get<0>(key)];
            int& dst = (std::get<1>(key) == 0) ? u.g3d : u.cuda;
            dst = (std::max)(dst, (int)std::lround(std::clamp(v, 0.0, 100.0)));
        }
        valid_ = true;
        return out;
    }

private:
    bool open()
    {
        if (query_) return true;
        if (failed_) return false;
        if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) { query_ = nullptr; failed_ = true; return false; }
        if (PdhAddEnglishCounterW(query_, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter_) != ERROR_SUCCESS) {
            PdhCloseQuery(query_); query_ = nullptr; failed_ = true; return false;
        }
        PdhCollectQueryData(query_);   // 使用率は 2 回の差分なので 1 回目を取っておく
        return true;
    }
    PDH_HQUERY   query_   = nullptr;
    PDH_HCOUNTER counter_ = nullptr;
    bool         failed_  = false;
    bool         valid_   = false;
};

// CUDA デバイスの LUID（無ければ false）。結果は覚えておく
bool cudaLuid(int deviceID, Luid* out)
{
    static std::map<int, std::pair<bool, Luid>> cache;
    auto it = cache.find(deviceID);
    if (it == cache.end()) {
        cudaDeviceProp prop{};
        std::pair<bool, Luid> r{false, {0, 0}};
        if (cudaGetDeviceProperties(&prop, deviceID) == cudaSuccess && prop.luidDeviceNodeMask != 0) {
            unsigned lo = 0, hi = 0;
            std::memcpy(&lo, prop.luid, 4);       // LUID { DWORD LowPart; LONG HighPart; }
            std::memcpy(&hi, prop.luid + 4, 4);
            r = {true, {hi, lo}};
        }
        it = cache.emplace(deviceID, r).first;
    }
    *out = it->second.second;
    return it->second.first;
}
} // namespace
#endif

info_thread::info_thread(QObject *parent)
    : QThread(parent) {
}

info_thread::~info_thread() {
    nvmlShutdown();
}

void info_thread::run() {
    // NVML 初期化
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        qDebug() << "Failed to initialize NVML:" << nvmlErrorString(result);
        return;
    }

    QTimer timer;
    connect(&timer, &QTimer::timeout, this, &info_thread::check_gpu_usage, Qt::DirectConnection);
    timer.setInterval(1000);
    timer.start();
    fflush(stdout);
    exec();
    fflush(stdout);
    timer.stop();
    timer.disconnect();
}

// CUDA のデバイス番号と NVML の番号は並び順が違うことがある
//   （CUDA は既定で速い順、NVML は PCI 順）。PCI バス ID で同じ GPU を引く
static bool nvmlHandleFor(const GPUInfo& g, nvmlDevice_t* out)
{
    char busId[32];
    std::snprintf(busId, sizeof(busId), "%08x:%02x:%02x.0", g.pciDomain, g.pciBus, g.pciDevice);
    if (nvmlDeviceGetHandleByPciBusId_v2(busId, out) == NVML_SUCCESS) return true;
    return nvmlDeviceGetHandleByIndex(g.deviceID, out) == NVML_SUCCESS;   // 予備
}

// 1 秒ごとに g_GPUInfo の使用率・メモリを更新する
//   GPU 全体・NVENC・NVDEC・メモリは NVML、3D・CUDA は PDH（Windows のみ。取れなければ -1）
//   GPU_Usage はタスクマネージャーと同じく各エンジンの最大
void info_thread::check_gpu_usage() {
#ifdef _WIN32
    static GpuEngineCounter engines;   // このスレッドだけが使う（info_thread は 1 つ）
    const std::map<Luid, EngineUsage> engineUsage = engines.sample();
#endif
    for (size_t i = 0; i < g_GPUInfo.size(); i++) {
        nvmlDevice_t device;
        if (!nvmlHandleFor(g_GPUInfo[i], &device)) {
            qDebug() << "Failed to get device handle for" << g_GPUInfo[i].deviceName;
            continue;   // 1 枚失敗しても他の GPU は更新する
        }

        // 取れなかった項目は 0 扱い（古い GPU / ドライバでも止まらないように）
        nvmlUtilization_t utilization{};
        unsigned int encUtil = 0, encSampling = 0;
        unsigned int decUtil = 0, decSampling = 0;
        nvmlMemory_t memInfo{};
        nvmlDeviceGetUtilizationRates(device, &utilization);
        nvmlDeviceGetEncoderUtilization(device, &encUtil, &encSampling);
        nvmlDeviceGetDecoderUtilization(device, &decUtil, &decSampling);
        const bool memOk = nvmlDeviceGetMemoryInfo(device, &memInfo) == NVML_SUCCESS;

        int use3d = -1, useCuda = -1;
#ifdef _WIN32
        Luid luid;
        if (cudaLuid(g_GPUInfo[i].deviceID, &luid)) {
            const auto it = engineUsage.find(luid);
            if (it != engineUsage.end()) {
                use3d   = it->second.g3d;
                useCuda = it->second.cuda;   // CUDA のエンジンが見つからない環境は -1（隠す）
            }
            // 3D は全 GPU にあるエンジン。誰も使っていなければインスタンスが無いので 0
            if (engines.valid() && use3d < 0) use3d = 0;
        }
#endif
        g_GPUInfo[i].Graphics3D_Usage = use3d;
        g_GPUInfo[i].Cuda_Usage       = useCuda;
        // 全体 = タスクマネージャーと同じく、各エンジンのうち最大
        g_GPUInfo[i].GPU_Usage     = (std::max)({ (int)(std::max)({ utilization.gpu, encUtil, decUtil }), use3d, useCuda });
        g_GPUInfo[i].Encoder_Usage = (int)encUtil;
        g_GPUInfo[i].Decoder_Usage = (int)decUtil;
        if (memOk)
            g_GPUInfo[i].Memory_Usage = (int)(memInfo.used / (1024 * 1024)); // MB
    }
}
