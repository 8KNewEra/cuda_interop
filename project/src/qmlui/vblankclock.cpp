#include "src/qmlui/vblankclock.h"
#include "src/main/__global__.h"

#include <QWindow>
#include <QDebug>
#include <cmath>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_2.h>
#endif

VblankClock& VblankClock::instance()
{
    static VblankClock c;
    return c;
}

VblankClock::~VblankClock()
{
    stop();
}

void VblankClock::setWindow(QWindow* window)
{
#ifdef _WIN32
    if (!window) return;
    hwnd_.store(quintptr(window->winId()));
    if (!running_.exchange(true))
        thread_ = std::thread([this] { run(); });
#else
    Q_UNUSED(window);
#endif
}

void VblankClock::stop()
{
    if (running_.exchange(false) && thread_.joinable())
        thread_.join();   // WaitForVBlank は次の vblank で戻る（出力がないときは 50ms で回る）
}

bool VblankClock::grid(qint64& vbNs, double& perNs) const
{
    if (samples_.load(std::memory_order_acquire) < 8) return false;
    vbNs  = lastVblankNs_.load(std::memory_order_acquire);
    perNs = double(periodPs_.load(std::memory_order_acquire)) / 1000.0;
    // 古すぎる（モニターが寝ている・測りが止まった）ときは使わない
    return vbNs > 0 && perNs >= 3e6 && perNs <= 50e6 && mono_now_ns() - vbNs < qint64(perNs * 4);
}

#ifdef _WIN32
namespace {
// モニターの DXGI 出力（全アダプタから探す。マルチ GPU でもモニターがつながっている GPU の出力が見つかる）
IDXGIOutput* findOutput(HMONITOR mon)
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
        return nullptr;
    IDXGIOutput* found = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; !found && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        IDXGIOutput* o = nullptr;
        for (UINT i = 0; !found && adapter->EnumOutputs(i, &o) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_OUTPUT_DESC d;
            if (SUCCEEDED(o->GetDesc(&d)) && d.Monitor == mon) { found = o; o = nullptr; }
            if (o) o->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return found;
}
} // namespace
#endif

void VblankClock::run()
{
#ifdef _WIN32
    HMONITOR     curMon = nullptr;
    IDXGIOutput* output = nullptr;
    qint64       prevNs = 0;
    double       perNs  = 0;
    int          checkCount = 0;

    while (running_.load()) {
        // モニターが変わったか（ウィンドウを移した）を時々見る
        if (!output || ++checkCount >= 30) {
            checkCount = 0;
            const HWND hwnd = HWND(hwnd_.load());
            const HMONITOR mon = hwnd ? MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) : nullptr;
            if (mon != curMon || !output) {
                if (output) { output->Release(); output = nullptr; }
                curMon = mon;
                output = mon ? findOutput(mon) : nullptr;
                prevNs = 0; perNs = 0;
                samples_.store(0, std::memory_order_release);
                if (output) {
                    DXGI_OUTPUT_DESC d;
                    output->GetDesc(&d);
                    qDebug() << "[VblankClock] 測るモニター:" << QString::fromWCharArray(d.DeviceName);
                }
            }
        }
        if (!output) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }

        if (FAILED(output->WaitForVBlank())) {
            output->Release(); output = nullptr;   // モードの切り替えなど。取り直す
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        const qint64 now = mono_now_ns();
        if (prevNs > 0) {
            const double dt = double(now - prevNs);
            if (dt >= 3e6 && dt <= 50e6) {
                if (perNs <= 0) perNs = dt;
                else {
                    // 1 周期ぶんの間隔だけ平均に入れる（取りこぼして 2 周期ぶん空いたものは入れない）
                    const double k = std::round(dt / perNs);
                    if (k >= 1 && k <= 3) perNs += (dt / k - perNs) * 0.02;
                }
                periodPs_.store(qint64(perNs * 1000.0), std::memory_order_release);
                lastVblankNs_.store(now, std::memory_order_release);
                const int s = samples_.load(std::memory_order_relaxed);
                if (s < 1000000) samples_.store(s + 1, std::memory_order_release);
            }
        }
        prevNs = now;
    }
    if (output) output->Release();
#endif
}
