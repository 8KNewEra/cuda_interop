#include "src/main/mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QStyleFactory>

std::vector<GPUInfo> g_GPUInfo;
AppSettings g_AppSettings{};
int g_openglDeviceID=0;
int g_EncodeRingNo = 0;
int g_EncodeRingSize = 36;

int main(int argc, char *argv[])
{
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
    MainWindow w;
    w.show();
    return app.exec();
}
