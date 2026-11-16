#include "src/qmlui/displayverify.h"
#include "src/qmlui/videoitem.h"

#include <QQuickWindow>
#include <QQuickItem>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QShortcut>
#include <QKeySequence>
#include <QTimer>
#include <QDebug>
#include <QSet>
#include <QRect>
#include <vector>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX   // windows.h の min / max マクロを止める（std::max と衝突する）
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#endif

namespace {

// 読み取る範囲（ウィンドウの物理画素。y は上から）
struct Row {
    int x0 = 0, x1 = 0;   // [x0, x1)
    int y  = 0;
    bool valid() const { return x1 > x0; }
};

struct Stats {
    int    levels  = 0;     // 異なる値の数
    double offGrid = -1;    // 8bit の格子に乗らない値の割合（%）。-1 = 判定しない
    double avgRun  = 0;     // 同じ値が続く画素数の平均
};

// 値の並び（ここでは 1 チャンネル）から段数・8bit 外の割合・平均幅を出す
//   offGridOf: 0..1023 の値なら 8bit の格子（round(round(v*255/1023)*1023/255)）と比べる
Stats analyze(const std::vector<int>& v, bool tenBitScale)
{
    Stats s;
    if (v.empty()) return s;
    QSet<int> uniq;
    int runs = 1, off = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        uniq.insert(v[i]);
        if (i > 0 && v[i] != v[i - 1]) ++runs;
        if (tenBitScale) {
            const int v8   = (v[i] * 255 + 511) / 1023;
            const int back = (v8 * 1023 + 127) / 255;
            if (back != v[i]) ++off;
        }
    }
    s.levels  = uniq.size();
    s.avgRun  = double(v.size()) / runs;
    s.offGrid = tenBitScale ? 100.0 * off / double(v.size()) : -1;
    return s;
}

QString describe(const char* name, const Stats& s, int width)
{
    QString t = QString("%1: 段数 %2 / 平均幅 %3px")
                    .arg(name).arg(s.levels).arg(s.avgRun, 0, 'f', 1);
    if (s.offGrid >= 0) t += QString(" / 8bit 外 %1%").arg(s.offGrid, 0, 'f', 1);
    t += QString("（%1px 読み取り）").arg(width);
    return t;
}

// 映像の表示枠（ウィンドウの論理座標）。VideoItem が見つからなければウィンドウ全体
QRectF videoRectInScene(QQuickWindow* w)
{
    const auto items = w->contentItem()->findChildren<VideoItem*>();
    for (VideoItem* it : items) {
        if (!it->isVisible() || !it->hasVideo()) continue;
        return it->mapRectToScene(it->videoRect());
    }
    return QRectF(0, 0, w->width(), w->height());
}

// カーソルの高さの 1 行（ウィンドウの物理画素）
Row pickRow(QQuickWindow* w)
{
    Row r;
    const qreal dpr = w->effectiveDevicePixelRatio();
    const QRectF vr = videoRectInScene(w).intersected(QRectF(0, 0, w->width(), w->height()));
    if (vr.isEmpty()) return r;
#ifdef _WIN32
    POINT p;
    GetCursorPos(&p);
    ScreenToClient(reinterpret_cast<HWND>(w->winId()), &p);   // 物理画素（Per-Monitor DPI aware）
    r.y = p.y;
#else
    r.y = int(w->height() * dpr / 2);
#endif
    r.x0 = int(std::ceil(vr.left() * dpr));
    r.x1 = int(std::floor(vr.right() * dpr));
    const int fbH = int(std::lround(w->height() * dpr));
    if (r.y < int(vr.top() * dpr) || r.y >= int(vr.bottom() * dpr) || r.y < 0 || r.y >= fbH)
        r.x1 = r.x0;   // 映像の外
    return r;
}

// ② 色バッファ（swap 前の back buffer）を読む。G チャンネルを 0..1023 で返す
std::vector<int> readFramebuffer(QQuickWindow* w, const Row& r)
{
    std::vector<int> out;
    QOpenGLContext* ctx = QOpenGLContext::currentContext();
    if (!ctx) return out;
    QOpenGLFunctions* f = ctx->functions();
    const int n   = r.x1 - r.x0;
    const int fbH = int(std::lround(w->height() * w->effectiveDevicePixelRatio()));
    std::vector<uint16_t> buf(size_t(n) * 4);
    w->beginExternalCommands();
    f->glPixelStorei(GL_PACK_ALIGNMENT, 1);
    f->glReadPixels(r.x0, fbH - 1 - r.y, n, 1, GL_RGBA, GL_UNSIGNED_SHORT, buf.data());
    w->endExternalCommands();
    out.resize(n);
    for (int i = 0; i < n; ++i)
        out[i] = int((uint32_t(buf[size_t(i) * 4 + 1]) * 1023u + 32767u) / 65535u);
    return out;
}

#ifdef _WIN32
float halfToFloat(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    float v;
    if (exp == 0)       v = std::ldexp(float(man), -24);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else                v = std::ldexp(float(man | 0x400), int(exp) - 25);
    return sign ? -v : v;
}

// ③ の途中経過（クラッシュしたときにどこまで進んだか分かるように 1 段ずつ出す）
void step(const char* s) { qDebug().noquote() << QString("[verify]   ③ %1").arg(QString::fromUtf8(s)); }

// ③ 画面合成（Desktop Duplication）を読む。G チャンネルを「段数を数えるための値」で返す
//   FP16（線形）    : ビット列そのまま（大小関係だけ使う。段数を数えるにはこれで十分）
//   R10G10B10A2     : 0..1023
//   B8G8R8A8 など   : 0..255
//   1 画素のバイト数は形式から決める（決め打ちで読むと範囲外を読んでクラッシュする）
//
//   作った直後の 1 枚目は中身が空（黒）のことがあるので使わない。ウィンドウを描き直させて、
//   「新しく描かれた」フレーム（LastPresentTime != 0）が来たら読む。GUI スレッドを止めないよう
//   QTimer で少しずつ待つ（止めるとこちらのウィンドウが描かれず、フレームが来ない）
class DesktopReader
{
public:
    QString err, fmtName;
    std::vector<int> values;

    ~DesktopReader()
    {
        if (dup) dup->Release();
        if (dc)  dc->Release();
        if (dev) dev->Release();
    }

    bool start(QQuickWindow* w)
    {
        step("出力を探す");
        hwnd = reinterpret_cast<HWND>(w->winId());
        const HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);

        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            err = "CreateDXGIFactory1 失敗"; return false;
        }
        IDXGIAdapter1* adapter = nullptr;
        IDXGIOutput5*  output5 = nullptr;
        for (UINT a = 0; !output5 && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
            IDXGIOutput* o = nullptr;
            for (UINT i = 0; !output5 && adapter->EnumOutputs(i, &o) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_OUTPUT_DESC d;
                if (SUCCEEDED(o->GetDesc(&d)) && d.Monitor == mon) {
                    o->QueryInterface(__uuidof(IDXGIOutput5), reinterpret_cast<void**>(&output5));
                    desk = d.DesktopCoordinates;
                }
                o->Release();
            }
            if (!output5) { adapter->Release(); adapter = nullptr; }
        }
        factory->Release();
        if (!output5) { err = "モニターの出力が見つからない（IDXGIOutput5 非対応）"; return false; }

        step("D3D11 デバイスを作る");
        HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                                       D3D11_SDK_VERSION, &dev, nullptr, &dc);
        adapter->Release();
        if (FAILED(hr)) { output5->Release(); err = "D3D11CreateDevice 失敗"; return false; }

        step("DuplicateOutput1");
        const DXGI_FORMAT fmts[] = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R10G10B10A2_UNORM,
                                     DXGI_FORMAT_B8G8R8A8_UNORM };
        hr = output5->DuplicateOutput1(dev, 0, UINT(std::size(fmts)), fmts, &dup);
        output5->Release();
        if (FAILED(hr)) {
            err = QString("DuplicateOutput1 失敗 (0x%1)").arg(quint32(hr), 8, 16, QChar('0'));
            return false;
        }
        return true;
    }

    // 新しく描かれたフレームが来ていれば読んで true。まだなら false（呼び出し側が描き直させて再試行）
    bool tryRead(const Row& r)
    {
        IDXGIResource* res = nullptr;
        DXGI_OUTDUPL_FRAME_INFO info = {};
        if (FAILED(dup->AcquireNextFrame(50, &info, &res)) || !res) return false;
        if (info.LastPresentTime.QuadPart == 0) {   // マウスだけの更新・空の 1 枚目
            res->Release();
            dup->ReleaseFrame();
            return false;
        }
        ID3D11Texture2D* tex = nullptr;
        res->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex));
        res->Release();
        if (tex) { readRow(tex, r); tex->Release(); }
        else     err = "画面のテクスチャを取得できない";
        dup->ReleaseFrame();
        return true;
    }

private:
    HWND hwnd = nullptr;
    RECT desk = {};
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* dc = nullptr;
    IDXGIOutputDuplication* dup = nullptr;

    void readRow(ID3D11Texture2D* tex, const Row& r)
    {
        // ウィンドウの行 → デスクトップ（このモニターの左上が 0）の座標
        POINT org = { 0, 0 };
        ClientToScreen(hwnd, &org);
        const int sx0 = org.x + r.x0 - desk.left;
        const int sx1 = org.x + r.x1 - desk.left;
        const int sy  = org.y + r.y  - desk.top;

        D3D11_TEXTURE2D_DESC td;
        tex->GetDesc(&td);
        int bpp = 0;   // 1 画素のバイト数
        switch (td.Format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: bpp = 8; fmtName = "FP16";          break;
        case DXGI_FORMAT_R10G10B10A2_UNORM:  bpp = 4; fmtName = "R10G10B10A2";   break;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UNORM:     bpp = 4; fmtName = "8bit（RGBA8）"; break;
        default: fmtName = QString("不明（DXGI_FORMAT %1）").arg(int(td.Format)); break;
        }
        step(QString("画面の形式: %1、%2×%3、行 y=%4 x=%5..%6（モニター内）")
                 .arg(fmtName).arg(td.Width).arg(td.Height).arg(sy).arg(sx0).arg(sx1 - 1).toUtf8().constData());
        const int x0 = (std::max)(0, sx0), x1 = (std::min)(int(td.Width), sx1);
        if (bpp == 0) { err = "対応していない画面の形式"; return; }
        if (x1 <= x0 || sy < 0 || sy >= int(td.Height)) { err = "読み取る行が画面の外"; return; }

        D3D11_TEXTURE2D_DESC sd = {};
        sd.Width = UINT(x1 - x0); sd.Height = 1; sd.MipLevels = 1; sd.ArraySize = 1;
        sd.Format = td.Format; sd.SampleDesc.Count = 1;
        sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &staging))) { err = "CreateTexture2D 失敗"; return; }
        D3D11_BOX box = { UINT(x0), UINT(sy), 0, UINT(x1), UINT(sy + 1), 1 };
        dc->CopySubresourceRegion(staging, 0, 0, 0, 0, tex, 0, &box);
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(dc->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
            const uint8_t* row = static_cast<const uint8_t*>(m.pData);
            const int n = x1 - x0;
            values.resize(size_t(n));
            for (int i = 0; i < n; ++i) {
                const uint8_t* p = row + size_t(i) * size_t(bpp);
                if (bpp == 8) {                                    // FP16: G は 2 個目の 16bit
                    values[size_t(i)] = int(uint16_t(p[2] | (p[3] << 8)));
                } else if (td.Format == DXGI_FORMAT_R10G10B10A2_UNORM) {
                    const uint32_t v = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
                    values[size_t(i)] = int((v >> 10) & 0x3FF);    // G
                } else {
                    values[size_t(i)] = p[1];                      // B8G8R8A8 / R8G8B8A8 とも G は 2 バイト目
                }
            }
            dc->Unmap(staging, 0);
            // 参考: 先頭と末尾の値（FP16 は線形 0..1 に直す）
            auto show = [&](int v) { return bpp == 8 ? QString::number(halfToFloat(uint16_t(v)), 'f', 5) : QString::number(v); };
            step(QString("先頭 %1 / 末尾 %2").arg(show(values.front()), show(values.back())).toUtf8().constData());
        } else {
            err = "Map 失敗";
        }
        staging->Release();
    }
};
#endif

} // namespace

void DisplayVerify::install(QQuickWindow* window)
{
    // 押されたら次のフレームの描画直後（swap 前）に読む
    static bool pending = false;
    static Row  row;

    auto* sc = new QShortcut(QKeySequence(QStringLiteral("Ctrl+Shift+D")), window);
    sc->setContext(Qt::ApplicationShortcut);
    QObject::connect(sc, &QShortcut::activated, window, [window] {
        row = pickRow(window);
        if (!row.valid()) {
            qDebug().noquote() << "[verify] マウスカーソルを映像の上に置いてから Ctrl+Shift+D を押してください";
            return;
        }
        pending = true;
        window->update();
    });

    QObject::connect(window, &QQuickWindow::afterRendering, window, [window] {
        if (!pending) return;
        pending = false;
        const Row r = row;
        const std::vector<int> fb = readFramebuffer(window, r);
        qDebug().noquote() << QString("[verify] 行 y=%1, x=%2..%3（ウィンドウの物理画素）").arg(r.y).arg(r.x0).arg(r.x1 - 1);
        qDebug().noquote() << "[verify] " + describe("② 色バッファ", analyze(fb, true), int(fb.size()));
#ifdef _WIN32
        // ③ は画面に出た後に読む。複製を作ってから描き直させ、新しいフレームが来るまで少しずつ待つ
        QTimer::singleShot(0, window, [window, r] {
            auto* rd = new DesktopReader;
            auto report = [rd] {
                if (rd->values.empty())
                    qDebug().noquote() << "[verify] ③ 画面合成: 読み取れません（" + rd->err + "）";
                else
                    qDebug().noquote() << "[verify] " + describe("③ 画面合成", analyze(rd->values, false), int(rd->values.size()))
                                          + "（形式 " + rd->fmtName + "）";
                delete rd;
            };
            if (!rd->start(window)) { report(); return; }
            step("新しいフレームを待つ");
            auto tries = std::make_shared<int>(0);
            auto poll = std::make_shared<std::function<void()>>();
            // 終わったら poll 自身への参照（循環）を外す。実行中の自分を消さないよう次のイベントで外す
            auto finish = [window, poll, report] {
                report();
                QTimer::singleShot(0, window, [poll] { *poll = nullptr; });
            };
            *poll = [window, r, rd, finish, tries, poll] {
                if (rd->tryRead(r)) { finish(); return; }
                if (++*tries >= 20) { rd->err = "新しいフレームが来ない（約 2 秒）"; finish(); return; }
                window->update();                                   // 描き直させてフレームを発生させる
                QTimer::singleShot(100, window, [poll] { if (*poll) (*poll)(); });
            };
            window->update();
            QTimer::singleShot(100, window, [poll] { if (*poll) (*poll)(); });
        });
#endif
    }, Qt::DirectConnection);

    qDebug().noquote() << "[verify] 表示の検証を有効化（Ctrl+Shift+D。カーソルの高さの 1 行を読む）";
}
