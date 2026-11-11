#include "src/main/mainwindow.h"
#include "src/qmlui/playercontroller.h"
#include "src/qmlui/videoengine.h"
#include "src/qmlui/videoitem.h"
#include "src/qmlui/windowchrome.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QQuickStyle>
#include <cstring>

std::vector<GPUInfo> g_GPUInfo;
AppSettings g_AppSettings{};
int g_openglDeviceID=0;

// ★ --qml で Qt Quick 版 UI、引数なしなら従来の Widgets 版。
//   環境変数 CUDA_VIEWER_UI=qml でも切り替えられる（Qt Creator の実行設定用）。
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
    // ★ VideoEngine のコンテキストとシーングラフのコンテキストで
    //   テクスチャを共有するために必須
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    // ★ 既定フォーマット。VideoEngine もシーングラフもこれで作る。
    //   ヒストグラムの描画が固定機能（glBegin など）を使うので Compatibility。
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
    fmt.setVersion(4, 5);
    fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
    fmt.setDepthBufferSize(24);
    fmt.setStencilBufferSize(8);
    fmt.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(fmt);

    // ★ Windows の Qt 6 は既定で D3D11 なので OpenGL を明示する
    QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);

    // ★ GUI スレッドで描画する（QOpenGLWindow 版と同じスレッド構成）。
    //   デコードとのハンドシェイク・CUDA interop がすべて GUI スレッドに
    //   揃うので、スレッド間の受け渡しを増やさずに済む。
    if (qEnvironmentVariableIsEmpty("QSG_RENDER_LOOP"))
        qputenv("QSG_RENDER_LOOP", "basic");

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

    // ★ player より後に宣言する（QML を先に破棄し、その後に engine を破棄する）
    // OS のタイトルバーを消してヘッダーをタイトルバーにする
    WindowChrome chrome;

    QQmlApplicationEngine qml;
    qml.rootContext()->setContextProperty("player", &player);
    qml.rootContext()->setContextProperty("chrome", &chrome);
    qml.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    if (qml.rootObjects().isEmpty())
        return -1;

    auto* window = qobject_cast<QQuickWindow*>(qml.rootObjects().first());
    if (window) {
        // ★ シーングラフが描き終えたことを engine に伝える（FBO 上書き前に待つ）
        QObject::connect(window, &QQuickWindow::frameSwapped, window, [] {
            if (VideoEngine* eng = VideoEngine::instance())
                eng->signalDisplayDone();
        }, Qt::DirectConnection);

        chrome.attach(window);

        // ダーク（タイトルバーは消しているが、システムメニュー等の配色用に残す）
        HWND hwnd = (HWND)window->winId();
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark)); // Windows 11
        DwmSetWindowAttribute(hwnd, 19, &dark, sizeof(dark)); // Windows 10 fallback
    }

    return app.exec();
}

int main(int argc, char *argv[])
{
    const bool qmlUi = useQmlUi(argc, argv);
    if (qmlUi)
        prepareQmlUi();

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
