#include "src/qmlui/windowchrome.h"

#include <QQuickWindow>
#include <QGuiApplication>
#include <QDebug>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "user32.lib")
#endif

WindowChrome::WindowChrome(QObject* parent)
    : QObject(parent)
{
}

WindowChrome::~WindowChrome()
{
    if (hwnd_ && qApp)
        qApp->removeNativeEventFilter(this);
}

bool WindowChrome::active() const
{
#ifdef Q_OS_WIN
    return true;
#else
    return false;
#endif
}

void WindowChrome::setTitleBar(QQuickItem* item)
{
    if (titleBar_ == item) return;
    titleBar_ = item;
    emit titleBarChanged();
}

void WindowChrome::attach(QQuickWindow* window)
{
#ifdef Q_OS_WIN
    if (!window) return;
    window_ = window;
    hwnd_ = reinterpret_cast<void*>(window->winId());
    HWND hwnd = static_cast<HWND>(hwnd_);

    qApp->installNativeEventFilter(this);

    // ★ 非クライアント領域を消しても DWM の影を残すため、1px だけ枠を
    //   クライアント領域に延ばしておく（中身で上書きされるので見えない）
    const MARGINS shadow = { 0, 0, 1, 0 };
    DwmExtendFrameIntoClientArea(hwnd, &shadow);

    // WM_NCCALCSIZE をもう一度出させて、タイトルバーを消した状態にする
    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                 SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE);
#else
    Q_UNUSED(window);
#endif
}

bool WindowChrome::isOverInteractive(QQuickItem* item, const QPointF& scenePos) const
{
    const auto children = item->childItems();
    // 手前（後ろの要素）から見る
    for (auto it = children.crbegin(); it != children.crend(); ++it) {
        QQuickItem* c = *it;
        if (!c->isVisible() || c->opacity() <= 0.0) continue;
        const QPointF p = c->mapFromScene(scenePos);
        const bool inside = c->contains(p);
        if (inside && c->acceptedMouseButtons() != Qt::NoButton)
            return true;
        // 子は親の外にはみ出していることがあるので、中に入らなくても探す
        if (isOverInteractive(c, scenePos))
            return true;
    }
    return false;
}

long WindowChrome::hitTest(long screenX, long screenY) const
{
#ifdef Q_OS_WIN
    HWND hwnd = static_cast<HWND>(hwnd_);
    if (!window_) return HTCLIENT;

    // 全画面中は普通のクライアント
    if (window_->visibility() == QWindow::FullScreen) return HTCLIENT;

    RECT wr;
    GetWindowRect(hwnd, &wr);
    const UINT dpi = GetDpiForWindow(hwnd);
    const int  bx  = GetSystemMetricsForDpi(SM_CXFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
    const int  by  = GetSystemMetricsForDpi(SM_CYFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);

    // ---- リサイズ枠（最大化中は無し）----
    if (!IsZoomed(hwnd)) {
        const bool left   = screenX <  wr.left + bx;
        const bool right  = screenX >= wr.right - bx;
        const bool top    = screenY <  wr.top + by;
        const bool bottom = screenY >= wr.bottom - by;
        if (top && left)     return HTTOPLEFT;
        if (top && right)    return HTTOPRIGHT;
        if (bottom && left)  return HTBOTTOMLEFT;
        if (bottom && right) return HTBOTTOMRIGHT;
        if (left)   return HTLEFT;
        if (right)  return HTRIGHT;
        if (top)    return HTTOP;
        if (bottom) return HTBOTTOM;
    }

    // ---- ヘッダー（ボタンの上以外はドラッグ移動）----
    if (titleBar_ && titleBar_->isVisible()) {
        POINT pt = { screenX, screenY };
        ScreenToClient(hwnd, &pt);
        const qreal dpr = window_->effectiveDevicePixelRatio();
        const QPointF scenePos(pt.x / dpr, pt.y / dpr);
        if (titleBar_->contains(titleBar_->mapFromScene(scenePos))) {
            if (isOverInteractive(titleBar_, scenePos)) return HTCLIENT;
            return HTCAPTION;
        }
    }
    return HTCLIENT;
#else
    Q_UNUSED(screenX); Q_UNUSED(screenY);
    return 0;
#endif
}

bool WindowChrome::nativeEventFilter(const QByteArray& eventType, void* message, qintptr* result)
{
#ifdef Q_OS_WIN
    if (eventType != "windows_generic_MSG" || !hwnd_) return false;
    MSG* msg = static_cast<MSG*>(message);
    if (msg->hwnd != static_cast<HWND>(hwnd_)) return false;
    HWND hwnd = msg->hwnd;

    switch (msg->message) {
    case WM_NCCALCSIZE: {
        if (msg->wParam != TRUE) break;
        if (window_ && window_->visibility() == QWindow::FullScreen) break;   // Qt に任せる

        // ★ タイトルバー（上辺）だけをクライアントにする。
        //   左右と下の枠は Windows 10/11 では透明なリサイズ用の帯で、影もここに
        //   出る。既定処理に任せてそのまま残す（Windows Terminal と同じやり方）。
        auto* p = reinterpret_cast<NCCALCSIZE_PARAMS*>(msg->lParam);
        const LONG originalTop = p->rgrc[0].top;
        const LRESULT r = DefWindowProcW(hwnd, WM_NCCALCSIZE, msg->wParam, msg->lParam);
        if (r != 0) { *result = r; return true; }
        p->rgrc[0].top = originalTop;

        // 最大化中はウィンドウが画面より枠のぶん上にはみ出しているので、
        // そのぶん下げないとヘッダーの上端が画面外に切れる
        if (IsZoomed(hwnd)) {
            const UINT dpi = GetDpiForWindow(hwnd);
            p->rgrc[0].top += GetSystemMetricsForDpi(SM_CYFRAME, dpi)
                            + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
        }
        *result = 0;
        return true;
    }

    case WM_NCHITTEST:
        *result = hitTest(GET_X_LPARAM(msg->lParam), GET_Y_LPARAM(msg->lParam));
        return true;

    case WM_NCACTIVATE:
        // ★ 非アクティブ化のたびに既定の枠を描き直させない（上端がちらつく）
        *result = DefWindowProcW(hwnd, WM_NCACTIVATE, msg->wParam, -1);
        return true;

    default:
        break;
    }
    return false;
#else
    Q_UNUSED(eventType); Q_UNUSED(message); Q_UNUSED(result);
    return false;
#endif
}
