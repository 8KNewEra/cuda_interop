#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "fps_thread.h"
#include "decode_thread.h"
#include "info_thread.h"
#define NOMINMAX

#include <QMainWindow>
#include <QVBoxLayout>
#include <windows.h>
#include "glwidget.h"

#include <windows.h>
#include <QMainWindow>
#include <QResizeEvent>
#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <QFileDialog>
#include <QShortcut>

extern int g_fps;

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
    void send_decode_speed(int speed);
    void send_manual_pause();
    void send_manual_resumeplayback();
    void send_manual_slider(int value);
    void decode_please();

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

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
    void start_decode_thread();
    void stop_decode_thread();
    void set_preview_speed(const QString &text);
    void decode_view(uint8_t* d_y, size_t pitch_y,uint8_t* d_uv, size_t pitch_uv,int height, int width);
    QThread *decode__thread;
    decode_thread *decodestream;
    bool run_decode_thread=false;
    int preview_speed=30;

    //スライダー
    void slider_control(int Frame_No);
    void slider_set_range(int pts,int maxframe,int framerate);
    int slider_max=0;
    int slider_min=1;
    int slider_No=1;
    int frame_pts=1;
    int framerate=33;

    //fpsスレッド
    void start_fps_thread();
    void fps_view();
    QThread *fps_view_thread;
    fps_thread *fpsstream;
    int fps_count=0;

    //使用率取得
    void start_info_thread();
    QThread *info_view_thread;
    info_thread *infostream;

    //ファイル
    void Open_Video_File();
    void Close_Video_File();
    QString input_filename;

    void gpu_encode();

private:
    Ui::MainWindow *ui;
    bool encode_flag=false;
};
#endif // MAINWINDOW_H

