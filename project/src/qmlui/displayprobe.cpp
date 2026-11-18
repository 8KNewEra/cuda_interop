#include "src/qmlui/displayprobe.h"
#include "src/qmlui/playercontroller.h"

#include <QQuickWindow>
#include <QTimer>
#include <QDebug>
#include <functional>
#include <iterator>
#include <memory>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX   // windows.h の min / max マクロを止める
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include "nvapi.h"                 // 3rdparty/nvapi（NVIDIA 公開の NVAPI SDK。MIT）
#include "NvApiDriverSettings.h"   // OGL_DEEP_COLOR_SCANOUT_ID
#endif

#ifdef _WIN32
namespace {

// ウィンドウがあるモニターの DXGI 出力（とそのアダプタ）。見つからなければ両方 nullptr
struct OutputRef {
    IDXGIAdapter1* adapter = nullptr;
    IDXGIOutput6*  output  = nullptr;
    ~OutputRef() { if (output) output->Release(); if (adapter) adapter->Release(); }
};

bool findOutput(HWND hwnd, OutputRef& ref)
{
    const HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
        return false;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; !ref.output && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        IDXGIOutput* o = nullptr;
        for (UINT i = 0; !ref.output && adapter->EnumOutputs(i, &o) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_OUTPUT_DESC d;
            if (SUCCEEDED(o->GetDesc(&d)) && d.Monitor == mon)
                o->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&ref.output));
            o->Release();
        }
        if (ref.output) ref.adapter = adapter;
        else            adapter->Release();
    }
    factory->Release();
    return ref.output != nullptr;
}

// モニターへの出力の深さ（NVIDIA の 8bpc / 10bpc 設定）。取れなければ 0
int outputBits(HWND hwnd)
{
    OutputRef ref;
    if (!findOutput(hwnd, ref)) return 0;
    DXGI_OUTPUT_DESC1 d = {};
    return SUCCEEDED(ref.output->GetDesc1(&d)) ? int(d.BitsPerColor) : 0;
}

// Desktop Duplication（画面合成の複製）。作れなければ dup == nullptr
struct Dup {
    ID3D11Device*           dev = nullptr;
    ID3D11DeviceContext*    dc  = nullptr;
    IDXGIOutputDuplication* dup = nullptr;
    ~Dup() { if (dup) dup->Release(); if (dc) dc->Release(); if (dev) dev->Release(); }

    bool create(HWND hwnd)
    {
        OutputRef ref;
        if (!findOutput(hwnd, ref)) return false;
        if (FAILED(D3D11CreateDevice(ref.adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                                     D3D11_SDK_VERSION, &dev, nullptr, &dc)))
            return false;
        // 合成が FP16 / 10bit ならその形式、通常の SDR なら 8bit で返ってくる
        const DXGI_FORMAT fmts[] = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R10G10B10A2_UNORM,
                                     DXGI_FORMAT_B8G8R8A8_UNORM };
        IDXGIOutput5* o5 = nullptr;
        if (FAILED(ref.output->QueryInterface(__uuidof(IDXGIOutput5), reinterpret_cast<void**>(&o5))))
            return false;
        const HRESULT hr = o5->DuplicateOutput1(dev, 0, UINT(std::size(fmts)), fmts, &dup);
        o5->Release();
        return SUCCEEDED(hr);
    }
};

int bitsOfFormat(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 16;
    case DXGI_FORMAT_R10G10B10A2_UNORM:  return 10;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UNORM:     return 8;
    default:                             return 0;
    }
}

// 画面合成の深さ（8 / 10 / 16）。取れなければ 0
//   最初の 1 枚（作った直後に来る今の画面）のテクスチャの形式を見る。来なければ複製の形式で代用
int compositionBits(HWND hwnd)
{
    Dup d;
    if (!d.create(hwnd)) return 0;
    IDXGIResource* res = nullptr;
    DXGI_OUTDUPL_FRAME_INFO info = {};
    int bits = 0;
    if (SUCCEEDED(d.dup->AcquireNextFrame(100, &info, &res)) && res) {
        ID3D11Texture2D* tex = nullptr;
        if (SUCCEEDED(res->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&tex)))) {
            D3D11_TEXTURE2D_DESC td;
            tex->GetDesc(&td);
            bits = bitsOfFormat(td.Format);
            tex->Release();
        }
        res->Release();
        d.dup->ReleaseFrame();
    }
    if (bits == 0) {
        DXGI_OUTDUPL_DESC od;
        d.dup->GetDesc(&od);
        bits = bitsOfFormat(od.ModeDesc.Format);
    }
    return bits;
}

// NVIDIA の「Deep color for 3D applications」（OGL_DEEP_COLOR_SCANOUT）がこのアプリに効く設定か
//   有効なら、ドライバが OpenGL の 10bit ウィンドウを画面合成（8bit）の外で直接 10bit でモニターへ出す。
//   Desktop Duplication からは見えないので、ドライバの設定（DRS）を読んで判断する。
//   このアプリ（exe のフルパス）のプロファイル → なければ全体（ベース）プロファイルの順。既定値は有効
//   戻り値: 1 = 有効 / 0 = 無効 / -1 = 読めない（NVIDIA 以外の GPU・古いドライバなど）
int nvDeepColorScanout()
{
    static int cached = -2;   // ドライバの設定は実行中ほぼ変わらないので 1 回だけ読む
    if (cached != -2) return cached;
    cached = -1;
    if (NvAPI_Initialize() != NVAPI_OK) return cached;

    NvDRSSessionHandle session = nullptr;
    if (NvAPI_DRS_CreateSession(&session) != NVAPI_OK) return cached;
    if (NvAPI_DRS_LoadSettings(session) == NVAPI_OK) {
        NvDRSProfileHandle profile = nullptr;

        // このアプリのプロファイル（フルパスで探すと、実行時にドライバが使うプロファイルが返る）
        NvAPI_UnicodeString exe = {};
        wchar_t path[MAX_PATH] = {};
        const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
        for (DWORD i = 0; i < n && i < NVAPI_UNICODE_STRING_MAX - 1; ++i) exe[i] = NvU16(path[i]);
        NVDRS_APPLICATION app = {};
        app.version = NVDRS_APPLICATION_VER;
        if (NvAPI_DRS_FindApplicationByName(session, exe, &profile, &app) != NVAPI_OK)
            profile = nullptr;
        if (!profile && NvAPI_DRS_GetBaseProfile(session, &profile) != NVAPI_OK)
            profile = nullptr;

        if (profile) {
            NVDRS_SETTING setting = {};
            setting.version = NVDRS_SETTING_VER;
            const NvAPI_Status st = NvAPI_DRS_GetSetting(session, profile, OGL_DEEP_COLOR_SCANOUT_ID, &setting);
            if (st == NVAPI_OK)
                cached = (setting.u32CurrentValue != OGL_DEEP_COLOR_SCANOUT_DISABLE) ? 1 : 0;
            else if (st == NVAPI_SETTING_NOT_FOUND)   // 誰も設定していない = 既定値（有効）
                cached = (OGL_DEEP_COLOR_SCANOUT_DEFAULT != OGL_DEEP_COLOR_SCANOUT_DISABLE) ? 1 : 0;
        }
    }
    NvAPI_DRS_DestroySession(session);
    return cached;
}

// 全画面で合成を通っているか（非同期）。描き直させながら約 0.4 秒、新しいフレームが来るか見る
//   来た = 合成を通っている（1）/ 来ない = 直接出している（0）/ 調べられない（-1）
void probeComposed(QQuickWindow* w, PlayerController* player)
{
    auto d = std::make_shared<Dup>();
    if (!d->create(reinterpret_cast<HWND>(w->winId()))) { player->setComposed(-1); return; }

    // 作った直後の 1 枚目（今の画面）は判定に使わない
    IDXGIResource* res = nullptr;
    DXGI_OUTDUPL_FRAME_INFO info = {};
    if (SUCCEEDED(d->dup->AcquireNextFrame(0, &info, &res)) && res) { res->Release(); d->dup->ReleaseFrame(); }

    auto tries = std::make_shared<int>(0);
    auto tick  = std::make_shared<std::function<void()>>();
    *tick = [w, player, d, tries, tick] {
        IDXGIResource* r = nullptr;
        DXGI_OUTDUPL_FRAME_INFO fi = {};
        bool presented = false;
        if (SUCCEEDED(d->dup->AcquireNextFrame(0, &fi, &r)) && r) {
            presented = fi.LastPresentTime.QuadPart != 0;   // 0 はマウスだけの更新
            r->Release();
            d->dup->ReleaseFrame();
        }
        const bool done = presented || ++*tries >= 8;
        if (done) {
            player->setComposed(presented ? 1 : 0);
            qDebug() << "[display] 全画面:" << (presented ? "画面合成を通る" : "直接出している（合成なし）");
            QTimer::singleShot(0, w, [tick] { *tick = nullptr; });   // 自分への参照（循環）を外す
            return;
        }
        w->update();   // 描き直させてフレームを発生させる
        QTimer::singleShot(50, w, [tick] { if (*tick) (*tick)(); });
    };
    w->update();
    QTimer::singleShot(50, w, [tick] { if (*tick) (*tick)(); });
}

} // namespace
#endif

void DisplayProbe::install(QQuickWindow* window, PlayerController* player)
{
#ifdef _WIN32
    // モニター出力と画面合成の深さ（イベントのときだけ読み直す）
    auto refresh = [window, player] {
        const HWND hwnd = reinterpret_cast<HWND>(window->winId());
        const int out  = outputBits(hwnd);
        const int comp = compositionBits(hwnd);
        const int deep = nvDeepColorScanout();
        player->setOutputBits(out);
        player->setCompositionBits(comp);
        player->setDeepColorScanout(deep == 1);
        qDebug() << "[display] モニター出力:" << out << "bit / 画面合成:" << comp << "bit"
                 << "/ NVIDIA Deep color:" << (deep == 1 ? "有効" : deep == 0 ? "無効" : "不明");
    };
    // 全画面なら合成を通るかを調べる（切り替え直後は落ち着くまで少し待つ）。ウィンドウ表示は常に通る
    auto checkFullScreen = [window, player] {
        if (window->visibility() != QWindow::FullScreen) { player->setComposed(1); return; }
        player->setComposed(-1);
        QTimer::singleShot(500, window, [window, player] {
            if (window->visibility() == QWindow::FullScreen) probeComposed(window, player);
        });
    };

    refresh();
    player->setComposed(1);
    QObject::connect(window, &QWindow::screenChanged, window, [refresh, checkFullScreen] { refresh(); checkFullScreen(); });
    QObject::connect(window, &QWindow::visibilityChanged, window, [refresh, checkFullScreen] { refresh(); checkFullScreen(); });
    QObject::connect(window, &QWindow::activeChanged, window, [window, refresh, checkFullScreen] {
        if (!window->isActive()) return;
        refresh();          // 設定（HDR / 自動カラー管理 / 出力の色深度）を変えて戻ってきたとき
        checkFullScreen();
    });
#else
    Q_UNUSED(window);
    Q_UNUSED(player);
#endif
}
