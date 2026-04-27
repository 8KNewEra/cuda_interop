#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "src/main/fps_thread.h"
#include "src/videoprocess/decode_thread.h"
#include "src/info/info_thread.h"
#include "src/info/logfile_control.h"
#include "src/ui_control/encode_setting.h"
#include "src/ui_control/jump_mode.h"
#include "src/ui_control/rangeslider.h"
#include "src/ui_control/video_speed.h"
#include "src/ui_control/audio_volume.h"
#define NOMINMAX

#include <QMainWindow>
#include <QVBoxLayout>
#include <windows.h>
#include "src/main/glwidget.h"
#include "src/main/__global__.h"

#include <QMainWindow>
#include <QResizeEvent>
#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <QFileDialog>
#include <QShortcut>
#include <QPointer>

//ドラッグアンドドロップ
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QFileInfo>
#include <QDebug>

#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

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
    void send_manual_back1frame();
    void send_manual_pause();
    void send_manual_reverse();
    void send_manual_resumeplayback();
    void send_manual_go1frame();
    void send_manual_slider(int value);
    void send_manual_range_end_slider(int value);
    void send_manual_range_start_slider(int value);
    void send_manual_high_res_slider(int value);
    void decode_please();

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    int p2p_test();
    void GLwidgetInitialized();
    QProgressDialog *progress;

    //画面表示
    int window_width=480;
    int window_height=360;
    void resizeEvent(QResizeEvent *event) override;
    void changeEvent(QEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *event)override;
    void toggleFullScreen();
    void CSS_Design();
    QIcon makeWhiteIcon(const QIcon &icon);

    //OpenGLwidget
    GLWidget* glWidget;
    QWidget *container;
    bool isFullScreenMode = false;

    //ライブスレッド動作
    void start_decode_thread(QString filePath);
    void stop_decode_thread();
    bool canUseGpuDecode(QString filename);
    void decode_view(VideoFrame Frame,bool pause,bool reverse);
    QThread *decode__thread=nullptr;
    decode_thread *decodestream=nullptr;
    bool run_decode_thread=false;
    int preview_speed=30;

    //スライダー
    void init_decodethread_complete();
    int slider_No=1;

    //再生・一時停止・スライダー
    void cutstart_pushbutton_control();
    void backstartframe_pushbutton_control();
    void back1frame_pushbutton_control();
    void reverse_pushbutton_control();
    void switch_resume_pause();
    void go1frame_pushbutton_control();
    void goendframe_pushbutton_control();
    void stop_pushbutton_control();
    void cutend_pushbutton_control();
    void slider_control(int value);
    void slider_start_control(int value);
    void slider_end_control(int value);
    void range_label_control(int range_time,int FrameNo);
    void heavy_process_UI_control(bool flag);
    void get_jump_value();
    RangeSlider *rangeSlider;
    e_jump_mode e_jumpmode = JUMP_MODE_SECOND;

    //再生、一時停止ステータス
    bool pause_flag = false;
    bool reverse_flag = false;
    int FrameNo = 0;

    //fpsスレッド
    void start_fps_thread(double target_fps);
    void stop_fps_thread();
    QThread *fps_view_thread=nullptr;
    fps_thread *fpsstream=nullptr;

    //使用率取得
    void start_info_thread();
    QThread *info_view_thread;
    info_thread *infostream;
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    //ファイル
    void Open_Video_File();
    void Close_Video_File();

    //エンコード
    void encode_set();
    void start_encode();
    void finished_encode();
    bool wasCanceled=false;
    encode_setting *encodeSetting;
    int encode_state=STATE_NOT_ENCODE;

    //オーディオ
    void init_async_audio();
    void play_audio(QByteArray pcm);
    QAudioSink* audioSink = nullptr;
    QIODevice* audioOutput = nullptr;

    audio_volume *audioVolume;
    bool audio_slider = false;

    //再生速度
    video_speed *videoSpeed;

    //Frameジャンプ、モード選択
    jump_mode *jumpMode;

    //iniファイル
    logfile_control *logFile;

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H

