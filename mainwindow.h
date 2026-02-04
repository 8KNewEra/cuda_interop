#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "fps_thread.h"
#include "decode_thread.h"
#include "info_thread.h"
#include "encode_setting.h"
#define NOMINMAX

#include <QMainWindow>
#include <QVBoxLayout>
#include <windows.h>
#include "glwidget.h"
#include "__global__.h"

#include <windows.h>
#include <QMainWindow>
#include <QResizeEvent>
#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <QFileDialog>
#include <QShortcut>
#include <QPointer>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

signals:
    void send_encode_flag(bool flag,int slider_max);
    void send_manual_pause();
    void send_manual_resumeplayback();
    void send_manual_slider(int value);
    void decode_please();

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void GLwidgetInitialized();
    QProgressDialog *progress;

    //画面表示
    int window_width=480;
    int window_height=360;
    void resizeEvent(QResizeEvent *event) override;
    void changeEvent(QEvent *event) override;
    void toggleFullScreen();

    //OpenGLwidget
    GLWidget* glWidget;
    QWidget *container;
    bool isFullScreenMode = false;

    //ライブスレッド動作
    void start_decode_thread(QString filePath);
    void stop_decode_thread();
    bool canUseGpuDecode(QString filename);
    void decode_view(VideoFrame Frame,bool pause_flag);
    QThread *decode__thread=nullptr;
    decode_thread *decodestream=nullptr;
    bool run_decode_thread=false;
    int preview_speed=30;

    //スライダー
    void slider_set_range();
    int slider_No=1;
    int Now_Frame;

    //fpsスレッド
    void start_fps_thread(double target_fps);
    void stop_fps_thread();
    QThread *fps_view_thread=nullptr;
    fps_thread *fpsstream=nullptr;

    //使用率取得
    void start_info_thread();
    QThread *info_view_thread;
    info_thread *infostream;

    //ファイル
    void Open_Video_File();
    void Close_Video_File();

    void encode_set();
    void start_encode();
    void finished_encode();
    encode_setting *encodeSetting;
    int encode_state=STATE_NOT_ENCODE;

    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //オーディオ
    void init_async_audio();
    void play_audio(QByteArray pcm);
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;
    bool audio_mode=false;

    //fpsタイマー
    QElapsedTimer fpsTimer;
    int fpsCount = 0;
    double fps = 0.0;

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H

