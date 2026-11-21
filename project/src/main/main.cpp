#include "src/main/mainwindow.h"
#include "src/qmlui/playercontroller.h"
#include "src/qmlui/videoengine.h"
#include "src/qmlui/videoitem.h"
#include "src/qmlui/sparklineitem.h"
#include "src/qmlui/vblankclock.h"
#include "src/qmlui/windowchrome.h"
#include "src/qmlui/encodecontroller.h"
#include "src/qmlui/screenshotsettings.h"
#include "src/qmlui/decodesettings.h"
#include "src/qmlui/displayprobe.h"
#ifdef CUDA_VIEWER_DISPLAY_VERIFY
#include "src/qmlui/displayverify.h"   // 検証用ビルドだけ（CONFIG+=display_verify）
#endif

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QQuickStyle>
#include <QOpenGLContext>
#include <QDebug>
#include <cstring>
#include <algorithm>


std::vector<GPUInfo> g_GPUInfo;
AppSettings g_AppSettings{};
int g_openglDeviceID=0;

// --qml で Qt Quick 版 UI、引数なしなら従来の Widgets 版
//   環境変数 CUDA_VIEWER_UI=qml でも切り替えられる（Qt Creator の実行設定用）
static bool useQmlUi(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--qml") == 0) return true;
        if (std::strcmp(argv[i], "--widgets") == 0) return false;
    }
    return qgetenv("CUDA_VIEWER_UI").toLower() == "qml";
}

// Qt Quick 版のための設定。QApplication を作る前に済ませること。
static void prepareQmlUi()
{
    // VideoEngine とシーングラフのコンテキストでテクスチャを共有するために必須
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    // 既定フォーマット。VideoEngine もシーングラフもこれで作る
    //   ヒストグラムの描画が固定機能（glBegin など）を使うので Compatibility
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
    fmt.setVersion(4, 5);
    fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
    fmt.setDepthBufferSize(24);
    fmt.setStencilBufferSize(8);
    // vsync（swap を vblank に合わせて待つか）は NVIDIA ドライバの「垂直同期」設定に合わせる（DisplayProbe::vsyncPolicy）。
    //   swap は描画スレッド（threaded render loop、下）で呼ぶので、vsync で swap が止まっても GUI スレッドは止まらない。
    //   オン: 描画スレッドが vblank ごとに swap し、表示のペース（docs/DECODE_CORE.md 3.6）がどのコマをどの vblank に出すかを決める。
    //   オフ・高速: 表示のペースは使わない（届いたらすぐ描く。ティアリングは覚悟。g_vsync_paced、__global__.h）。設定の変更は再起動で反映
    {
        const DisplayProbe::VsyncPolicy vp = DisplayProbe::vsyncPolicy();
        g_vsync_swap_interval = vp.swapInterval;
        g_vsync_paced         = vp.paced;
        g_vsync_text          = vp.text;
        fmt.setSwapInterval(vp.swapInterval);
    }
    // 表示の 10bit 化: ウィンドウの色バッファを 10:10:10 で要求する（10bit の動画は RGBA16 のまま
    //   シーングラフが描くので、ここが 10bit なら 10bit のまま画面に出る）。
    //   取れない環境（GPU・ドライバ・モニタが非対応）では 8bit の形式に落ちるだけで、動作は変わらない。
    //   実際に取れた深さは動画情報の「Process/Render Depth」の後ろ側に出す（PlayerController::setDisplayBits）。
    //   環境変数 CUDA_VIEWER_DISPLAY_BITS=8 で従来の 8bit に戻せる（問題が出たとき用）
    //   VideoEngine・共有コンテキスト・ウィンドウはすべてこの既定フォーマットで作る（共有には同じ形式が必要）
    if (qgetenv("CUDA_VIEWER_DISPLAY_BITS") != "8") {
        fmt.setRedBufferSize(10);
        fmt.setGreenBufferSize(10);
        fmt.setBlueBufferSize(10);
        // アルファも 2bit で指定する（10:10:10:2）。RGB だけだと WGL が 8bit の形式を選んでしまう
        //   （アルファ 8bit を残す形式が優先される）。実機（NVIDIA）で確認済み
        fmt.setAlphaBufferSize(2);
    }
    QSurfaceFormat::setDefaultFormat(fmt);

    // Windows の Qt 6 は既定で D3D11 なので OpenGL を明示する
    QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);

    // 描画スレッドで描画する（threaded render loop）。
    //   以前は GUI スレッドで描いていた（basic）。vsync オンのウィンドウ表示では swap が前のコマが画面に出るまで止まることがあり、
    //   その間 GUI スレッドのコマの受け取り・upload・デコードへの返事も止まって、目標がリフレッシュ以上のとき（60Hz で 2 倍速など）
    //   54〜56fps に張り付いていた。描画スレッドに分ければ止まるのは描画スレッドだけになる。
    //   engine（CUDA interop・GLSL の処理）は GUI スレッドの自前のコンテキストのまま。描画スレッドとは表示の組を 2 つにして
    //   受け渡す（videoengine.h の冒頭）。QSG_RENDER_LOOP=basic を指定すれば以前の描き方でも動く（表示のペースは粗くなる）
    if (qEnvironmentVariableIsEmpty("QSG_RENDER_LOOP"))
        qputenv("QSG_RENDER_LOOP", "threaded");

    // カスタマイズ可能なスタイル（Windows スタイルは背景差し替えを嫌う）
    QQuickStyle::setStyle("Basic");
}

static int runQmlUi(QApplication& app)
{
    PlayerController player;
    if (!player.initialize()) {
        QMessageBox::critical(nullptr, QObject::tr("初期化エラー"),
                              QObject::tr("OpenGL / CUDA の初期化に失敗しました。\n"
                                          "Widgets 版（引数なし）で起動してください。"));
        return 1;
    }

    qmlRegisterType<VideoItem>("CudaViewer", 1, 0, "VideoItem");
    qmlRegisterType<HistogramItem>("CudaViewer", 1, 0, "HistogramItem");
    qmlRegisterType<SparklineItem>("CudaViewer", 1, 0, "SparklineItem");   // qml/Sparkline.qml の中身

    // player より後に宣言する（QML を先に破棄し、その後に engine を破棄する）
    // OS のタイトルバーを消してヘッダーをタイトルバーにする
    WindowChrome chrome;

    QQmlApplicationEngine qml;
    qml.rootContext()->setContextProperty("player", &player);
    qml.rootContext()->setContextProperty("chrome", &chrome);
    qml.rootContext()->setContextProperty("encoder", player.encoder());
    qml.rootContext()->setContextProperty("shotSettings", player.screenshotSettings());
    qml.rootContext()->setContextProperty("decodeSettings", player.decodeSettings());
    qml.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    if (qml.rootObjects().isEmpty())
        return -1;

    auto* window = qobject_cast<QQuickWindow*>(qml.rootObjects().first());
    if (window) {
        // sync を始めたことを engine に伝える（描画スレッド。拍が済んだ swap に応えて待たないため）
        QObject::connect(window, &QQuickWindow::beforeSynchronizing, window, [] {
            if (VideoEngine* eng = VideoEngine::instance())
                eng->noteBeforeSync();
        }, Qt::DirectConnection);
        // swap したことを engine に伝える（描画スレッド。表示の計測と拍）
        QObject::connect(window, &QQuickWindow::frameSwapped, window, [] {
            if (VideoEngine* eng = VideoEngine::instance())
                eng->onRenderThreadSwapped();
        }, Qt::DirectConnection);

        chrome.attach(window);

        // 表示のペース: 出してよい時刻を待つコマがあるとき、swap の直後にこのウィンドウの sync をその場で進める（VideoEngine::kickPresent）
        if (VideoEngine* eng = VideoEngine::instance())
            eng->setPresentWindow(window);
        // 表示のペース: ウィンドウがあるモニターの vblank を測る（DWM は画面全体＝メインのモニターの値しか返さない）
        VblankClock::instance().setWindow(window);
        // 描き終えたことを engine に伝える（描画スレッド。swap の前）: 表から外した組に書く前に待つフェンスと、
        //   描き終えてから swap が戻るまでの時間（swap が止まっていないか）の起点
        QObject::connect(window, &QQuickWindow::afterRendering, window, [] {
            if (VideoEngine* eng = VideoEngine::instance())
                eng->noteAfterRendering();
        }, Qt::DirectConnection);

#ifdef CUDA_VIEWER_DISPLAY_VERIFY
        DisplayVerify::install(window);   // 表示の 10bit 検証（Ctrl+Shift+D）。検証用ビルドだけ
#endif

        // 実際に取れた表示の色深度（10 / 8）を動画情報へ。コンテキストができた時点の値を読む
        //   sceneGraphInitialized は描画スレッドで来るので、player へは GUI スレッドで渡す
        auto reportBits = [window, &player] {
            QOpenGLContext* c = QOpenGLContext::currentContext();
            const int bits = c ? c->format().redBufferSize() : window->format().redBufferSize();
            QMetaObject::invokeMethod(&player, [&player, bits] { player.setDisplayBits(bits); }, Qt::QueuedConnection);
            qDebug() << "[main] 表示の色深度:" << bits << "bit";
        };
        if (window->isSceneGraphInitialized()) reportBits();
        else QObject::connect(window, &QQuickWindow::sceneGraphInitialized, window, reportBits, Qt::DirectConnection);

        // 画面に出るまでの深さ（画面合成・全画面で合成を通るか・モニター出力）を調べて動画情報へ
        DisplayProbe::install(window, &player);

        // ダーク（タイトルバーは消しているが、システムメニュー等の配色用に残す）
        HWND hwnd = (HWND)window->winId();
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark)); // Windows 11
        DwmSetWindowAttribute(hwnd, 19, &dark, sizeof(dark)); // Windows 10 fallback
    }

    const int rc = app.exec();
    VblankClock::instance().stop();
    return rc;
}

int main(int argc, char *argv[])
{
    const bool qmlUi = useQmlUi(argc, argv);
    if (qmlUi)
        prepareQmlUi();
    g_AppSettings.gui_reads_packed_tiles = qmlUi;   // Widgets 版（GLWidget）は詰めたタイルを読めない

    QApplication app(argc, argv);

    qRegisterMetaType<QByteArray>();
    qRegisterMetaType<QByteArray>("QByteArray");
    app.setStyle(QStyleFactory::create("Fusion"));

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "realtime_render_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            app.installTranslator(&translator);
            break;
        }
    }

    if (qmlUi)
        return runQmlUi(app);

    MainWindow w;
    w.show();
    return app.exec();
}
