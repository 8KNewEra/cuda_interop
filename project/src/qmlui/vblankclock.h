#ifndef VBLANKCLOCK_H
#define VBLANKCLOCK_H

// ==============================================================
// VblankClock（ウィンドウがあるモニターの vblank の時計）
//
//   表示のペース（PlayerController。docs/DECODE_CORE.md 3.6）は「直近の vblank の時刻」と「周期」から
//   コマを出す vblank を選ぶ。以前は DWM（DwmGetCompositionTimingInfo）の値を使っていたが、Windows 8.1 以降は
//   hwnd を指定できず画面合成全体（メインのモニター）の値しか返さない。メインが 120Hz でウィンドウが 60Hz のモニターに
//   あると、120Hz の格子で出す時刻を決めてしまい、60Hz の vblank とずれて swap が毎回止まり、
//   目標がリフレッシュを超える再生で 55fps 前後に落ちていた。
//
//   ここでは専用のスレッドが、ウィンドウがあるモニターの DXGI 出力で IDXGIOutput::WaitForVBlank を回し、
//   vblank のたびに時刻（mono_now_ns。__global__.h）を記録する。周期は vblank の間隔の移動平均。
//   ウィンドウが別のモニターへ移ったら（MonitorFromWindow が変わったら）出力を取り直す。
//   出力が見つからない・測れていない間は grid() が false を返し、呼び出し側は DWM の値に戻る
// ==============================================================

#include <QtGlobal>
#include <atomic>
#include <thread>

class QWindow;

class VblankClock
{
public:
    static VblankClock& instance();

    // 測るウィンドウ（GUI スレッドから。初回でスレッドを起こす）
    void setWindow(QWindow* window);
    // 止める（アプリの終了前。GUI スレッドから）
    void stop();

    // 直近の vblank の時刻（mono_now_ns）と周期（ns）。測れていなければ false
    bool grid(qint64& vbNs, double& perNs) const;

private:
    VblankClock() = default;
    ~VblankClock();
    void run();

    std::thread           thread_;
    std::atomic<bool>     running_{false};
    std::atomic<quintptr> hwnd_{0};
    std::atomic<qint64>   lastVblankNs_{0};
    std::atomic<qint64>   periodPs_{0};     // 周期（ピコ秒。atomic に double を置かないため）
    std::atomic<int>      samples_{0};      // 今の出力で測った vblank の数（少ないうちは使わない）
};

#endif // VBLANKCLOCK_H
