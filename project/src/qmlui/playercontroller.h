#ifndef PLAYERCONTROLLER_H
#define PLAYERCONTROLLER_H

// ==============================================================
// PlayerController（QML 版の中心。context property "player"）
//
//   Widgets 版 MainWindow の「UI 以外の部分」を QML から使える形にしたもの。
//   デコード／fps／info スレッドの管理、再生操作、範囲、ジャンプ、
//   フィルタ、エンコードの段取りは MainWindow と同じ手順。
//
//   映像そのものは VideoEngine → VideoItem が直接やり取りするので、
//   ここを通るのはフレーム番号や時刻などの軽い値だけ。
//   画像処理の設定は g_AppSettings に直接持つ（終了時に logfile_control が ini 保存）。
//   変更したら engine の filter_change_flag を立てて再描画する。
// ==============================================================

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVariantMap>
#include <QVariantList>
#include <QElapsedTimer>
#include "src/main/__global__.h"

class VideoEngine;
class decode_thread;
class fps_thread;
class info_thread;
class logfile_control;
class EncodeController;
class FilterKernel;
class MediaProbe;
class ScreenshotSettings;

class PlayerController : public QObject
{
    Q_OBJECT

    // ---- 状態 ----
    Q_PROPERTY(bool ready           READ ready           NOTIFY stateChanged)
    Q_PROPERTY(bool fileOpen        READ fileOpen        NOTIFY stateChanged)
    Q_PROPERTY(bool controlsEnabled READ controlsEnabled NOTIFY stateChanged)
    Q_PROPERTY(bool playing         READ playing         NOTIFY stateChanged)
    Q_PROPERTY(bool reversing       READ reversing       NOTIFY stateChanged)
    Q_PROPERTY(bool encoding        READ encoding        NOTIFY stateChanged)
    // 連続して再生している間だけ true（コマ送り 1 枚では立たない）
    //   描画フレームレートの表示はこれを見る
    Q_PROPERTY(bool renderActive    READ renderActive    NOTIFY stateChanged)

    // ---- フレーム・時刻 ----
    Q_PROPERTY(int     frameNo    READ frameNo    NOTIFY frameChanged)
    Q_PROPERTY(QString playTime   READ playTime   NOTIFY frameChanged)
    Q_PROPERTY(int     maxFrames  READ maxFrames  NOTIFY stateChanged)
    Q_PROPERTY(int     rangeStart READ rangeStart NOTIFY rangeChanged)
    Q_PROPERTY(int     rangeEnd   READ rangeEnd   NOTIFY rangeChanged)
    Q_PROPERTY(QString startTime  READ startTime  NOTIFY rangeChanged)
    Q_PROPERTY(QString endTime    READ endTime    NOTIFY rangeChanged)
    Q_PROPERTY(QString rangeTime  READ rangeTime  NOTIFY rangeChanged)
    Q_PROPERTY(int     rangeFrames READ rangeFrames NOTIFY rangeChanged)

    // ---- 表示 ----
    Q_PROPERTY(bool infoVisible READ infoVisible WRITE setInfoVisible NOTIFY displayChanged)
    Q_PROPERTY(bool histVisible READ histVisible WRITE setHistVisible NOTIFY displayChanged)

    // ---- 画像処理（GLSL）----
    Q_PROPERTY(bool gaussian  READ gaussian  WRITE setGaussian  NOTIFY filterChanged)
    // 係数（サイズ・σ・直接編集）。中身は FilterKernel を参照
    Q_PROPERTY(QObject* gaussianKernel  READ gaussianKernel  CONSTANT)
    Q_PROPERTY(QObject* averagingKernel READ averagingKernel CONSTANT)
    // 動画ファイルの詳細情報（プロパティ画面）
    Q_PROPERTY(QObject* mediaInfo READ mediaInfo CONSTANT)
    Q_PROPERTY(bool sobel     READ sobel     WRITE setSobel     NOTIFY filterChanged)
    Q_PROPERTY(bool averaging READ averaging WRITE setAveraging NOTIFY filterChanged)
    Q_PROPERTY(bool zebra     READ zebra     WRITE setZebra     NOTIFY filterChanged)
    Q_PROPERTY(bool imageAdjust  READ imageAdjust  WRITE setImageAdjust  NOTIFY filterChanged)   // 画像補正
    Q_PROPERTY(bool colorCorrect READ colorCorrect WRITE setColorCorrect NOTIFY filterChanged)   // 色補正（R/G/B・彩度）
    Q_PROPERTY(bool channelView  READ channelView  WRITE setChannelView  NOTIFY filterChanged)
    Q_PROPERTY(bool compare      READ compare      WRITE setCompare      NOTIFY filterChanged)   // 比較表示
    Q_PROPERTY(bool tileSeam     READ tileSeam     WRITE setTileSeam     NOTIFY filterChanged)   // タイル境界チェック
    Q_PROPERTY(bool eyedropper   READ eyedropper   WRITE setEyedropper   NOTIFY filterChanged)   // スポイト
    // スポイトの最新の値 {frame, x, y, r, g, b, r0, g0, b0}。描画したフレームごとに更新
    Q_PROPERTY(QVariantMap pick  READ pick  NOTIFY pickChanged)
    Q_PROPERTY(bool toneCurve    READ toneCurve    WRITE setToneCurve    NOTIFY filterChanged)   // トーンカーブ
    // 4 本のカーブの点 [[{x,y},...] ×4]（0 = RGB 全体、1 = R、2 = G、3 = B）
    Q_PROPERTY(QVariantList curves READ curves NOTIFY filterChanged)
    // 各ツールの細かいパラメータ（キー名 → 値）。setFilterParam(key, v) で変更
    //   キー一覧と範囲は playercontroller.cpp の setFilterParam を参照
    Q_PROPERTY(QVariantMap filterParams READ filterParams NOTIFY filterChanged)

    // ---- 再生設定 ----
    Q_PROPERTY(int    jumpMode  READ jumpMode  WRITE setJumpMode  NOTIFY jumpChanged)
    Q_PROPERTY(int    jumpValue READ jumpValue WRITE setJumpValue NOTIFY jumpChanged)
    Q_PROPERTY(double speed     READ speed     WRITE setSpeed     NOTIFY speedChanged)
    Q_PROPERTY(int    volume    READ volume    WRITE setVolume    NOTIFY volumeChanged)

    // ---- 情報表示（infoChanged は動画情報の表示中・エンコード中以外に 500ms ごと）----
    //   videoInfo: ファイル・コーデック・解像度・pixel（ビット深度など）・audio（コーデック等）・fps など
    //   gpus: [{id, name, sm, usage, g3d, cuda, dec, memUsed, memMax, primary}]（g3d / cuda は取れない環境で -1）
    Q_PROPERTY(QVariantMap  videoInfo READ videoInfo NOTIFY infoChanged)
    Q_PROPERTY(QVariantList gpus      READ gpus      NOTIFY infoChanged)
    Q_PROPERTY(QVariantMap  histogram READ histogram NOTIFY histogramChanged)

public:
    explicit PlayerController(QObject* parent = nullptr);
    ~PlayerController() override;

    // GL / CUDA の初期化。QML を読み込む前に呼ぶ
    bool initialize();

    // エンコード設定（QML には "encoder" として公開する）
    QObject* gaussianKernel() const;
    QObject* averagingKernel() const;
    QObject* mediaInfo() const;
    Q_INVOKABLE void showProperties();     // 開いているファイルの詳細情報を読んで画面を出す
    EncodeController* encoder() const { return encoder_; }
    ScreenshotSettings* screenshotSettings() const { return shotSettings_; }

    bool ready() const           { return ready_; }
    bool fileOpen() const        { return run_decode_thread; }
    bool controlsEnabled() const { return controls_enabled_; }
    bool playing() const         { return run_decode_thread && !pause_flag; }
    bool reversing() const       { return reverse_flag; }
    bool encoding() const        { return encode_state != STATE_NOT_ENCODE; }
    bool renderActive() const    { return render_active_; }

    // 高精度シーク中は、デコードが目標に着くまで目標の番号を見せる
    //   （途中のコマや古いコマの番号で表示が揺れないように）
    int     frameNo() const    { return seek_hold_ ? seek_target_ : FrameNo; }
    // 時刻も開始・終了と同じくフレーム番号から出す（シーク保持の解除前後で
    //   計算方法が変わると、同じコマでも秒の繰り上がりがずれて表示が揺れる）
    QString playTime() const   { return formatFrames(frameNo()); }
    int     maxFrames() const;
    int     rangeStart() const { return range_start_; }
    int     rangeEnd() const   { return range_end_; }
    QString startTime() const  { return formatFrames(range_start_); }
    QString endTime() const    { return formatFrames(range_end_); }
    QString rangeTime() const  { return formatFrames(range_end_ - range_start_); }
    int     rangeFrames() const { return range_end_ - range_start_; }

    bool infoVisible() const { return g_AppSettings.videoInfo_flag; }
    bool histVisible() const { return g_AppSettings.histgram_flag; }
    void setInfoVisible(bool on);
    void setHistVisible(bool on);

    bool gaussian() const  { return g_AppSettings.gaussianfilterEnabled; }
    bool sobel() const     { return g_AppSettings.sobelfilterEnabled; }
    bool averaging() const { return g_AppSettings.averagingfilterEnabled; }
    void setGaussian(bool on);
    void setSobel(bool on);
    void setAveraging(bool on);
    bool zebra() const { return g_AppSettings.zebraEnabled; }
    void setZebra(bool on);
    bool imageAdjust() const  { return g_AppSettings.colorEnabled; }
    void setImageAdjust(bool on);
    Q_INVOKABLE void resetImageAdjust();    // 明るさ〜ガンマを既定値へ
    bool colorCorrect() const { return g_AppSettings.rgbEnabled; }
    bool channelView() const  { return g_AppSettings.channelEnabled; }
    bool compare() const      { return g_AppSettings.compareEnabled; }
    bool tileSeam() const     { return g_AppSettings.seamEnabled; }
    bool eyedropper() const   { return g_AppSettings.eyedropperEnabled; }
    void setEyedropper(bool on);
    QVariantMap pick() const;
    // スポイト: 映像上の位置（0..1、左上原点。範囲外で停止）を渡し、最後に拾えた値を返す
    Q_INVOKABLE QVariantMap pickPixel(double nx, double ny);
    void setTileSeam(bool on);
    bool toneCurve() const    { return g_AppSettings.curveEnabled; }
    void setToneCurve(bool on);
    QVariantList curves() const;
    Q_INVOKABLE void setCurvePoints(int ch, const QVariantList& pts);   // [{x,y},...]
    Q_INVOKABLE void resetCurve(int ch);
    Q_INVOKABLE void resetAllCurves();
    Q_INVOKABLE QVariantList curveLut(int ch) const;   // 描画用（そのカーブ単体の 256 段、0..255）
    void setCompare(bool on);
    void setChannelView(bool on);
    void setColorCorrect(bool on);
    Q_INVOKABLE void resetColorCorrect();   // R/G/B と彩度を既定値へ
    QVariantMap filterParams() const;
    Q_INVOKABLE void setFilterParam(const QString& key, const QVariant& value);

    int    jumpMode() const { return g_AppSettings.frame_jump_mode; }
    int    jumpValue() const;
    double speed() const    { return g_AppSettings.video_speed_ratio; }
    int    volume() const   { return g_AppSettings.audio_volume; }
    void setJumpMode(int mode);
    void setJumpValue(int value);
    void setSpeed(double ratio);
    void setVolume(int value);

    QVariantMap  videoInfo() const;
    QVariantList gpus() const;
    QVariantMap  histogram() const;

    // ---- QML から呼ぶ操作 ----
    Q_INVOKABLE void openFileDialog();
    Q_INVOKABLE void openFile(const QUrl& url);
    Q_INVOKABLE bool isSupportedFile(const QUrl& url) const;
    Q_INVOKABLE void closeFile();

    Q_INVOKABLE void togglePlay();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void reverse();
    Q_INVOKABLE void stepBack();       // ◀| ジャンプ量だけ戻る
    Q_INVOKABLE void stepForward();    // |▶ ジャンプ量だけ進む
    Q_INVOKABLE void toRangeStart();   // |◀
    Q_INVOKABLE void toRangeEnd();     // ▶|
    Q_INVOKABLE void cutStart();       // [
    Q_INVOKABLE void cutEnd();         // ]

    Q_INVOKABLE void scrub(int frame);          // スライダー操作中
    Q_INVOKABLE void scrubRelease(int frame);   // 離したとき（高精度シーク）
    Q_INVOKABLE void setRangeStartFrame(int frame);
    Q_INVOKABLE void setRangeEndFrame(int frame);
    // 範囲つまみを離したとき。つまみに押された再生位置を高精度シークで確定する
    Q_INVOKABLE void rangeDragFinished();

    Q_INVOKABLE void allFiltersOff();
    Q_INVOKABLE void screenshot();
    Q_INVOKABLE void openScreenshotSettings();
    Q_INVOKABLE void openEncodeSettings();

    Q_INVOKABLE QString jumpLabel(bool forward) const;   // "+3s" / "-1f" など

signals:
    void stateChanged();
    void frameChanged();
    void rangeChanged();
    void displayChanged();
    void filterChanged();
    void jumpChanged();
    void speedChanged();
    void volumeChanged();
    void infoChanged();
    void histogramChanged();
    void pickChanged();
    void screenshotStateChanged(bool busy);
    // エンコード設定を開いた（QML がパネルを出す）
    void encodeSettingsOpened();

    // decode_thread へ（MainWindow と同じ）
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

private:
    void start_decode_thread(const QString& filePath);
    void stop_decode_thread();
    bool canUseGpuDecode(const QString& filename);
    void decode_view(VideoFrame Frame, bool pause, bool reverse);
    void init_decodethread_complete();
    void setControlsEnabled(bool on);

    void start_fps_thread(double target_fps);
    void stop_fps_thread();
    void set_fps_driving(bool on);
    void start_info_thread();
    void stop_info_thread();

    void start_encode();
    void finished_encode();
    void encode_error_handler(const QString& error);

    void jumpBy(bool forward);
    QString formatFrames(int frames) const;

    // ---- 部品 ----
    VideoEngine*     engine_ = nullptr;
    logfile_control* logFile = nullptr;
    FilterKernel* gaussKernel_ = nullptr;   // ガウシアンの係数
    FilterKernel* avgKernel_   = nullptr;   // 平均化の係数
    MediaProbe*   mediaInfo_   = nullptr;   // プロパティ画面
    EncodeController* encoder_ = nullptr;   // エンコード設定（QML 版）
    ScreenshotSettings* shotSettings_ = nullptr;   // 画像保存設定（QML 版）

    decode_thread* decodestream = nullptr;
    QThread*       decode__thread = nullptr;
    QThread*       fps_view_thread = nullptr;
    fps_thread*    fpsstream = nullptr;
    info_thread*   infostream = nullptr;
    QTimer         infoTimer_;

    // ---- 状態 ----
    bool ready_ = false;
    bool run_decode_thread = false;
    bool controls_enabled_ = false;
    bool pause_flag = false;
    bool reverse_flag = false;
    // 一時停止でないフレームが続いた枚数（コマ送りと連続再生の区別用）
    int  steady_play_frames_ = 0;
    bool render_active_ = false;
    static const int kSteadyPlayFrames = 5;

    // ユーザーが再生（順・逆）を指示している間だけ true
    //   シークやコマ送りでも pause=false のフレームが流れるので、
    //   描画 fps の表示はフレームの状態ではなくこちらを条件にする
    bool play_intent_ = false;
    // エンコード設定を開いた時点の位置と再生状態（閉じたらここへ戻す）
    int  resume_frame_   = -1;
    bool resume_playing_ = false;
    void setPlayIntent(bool on);

    // 高精度シークの表示保持
    bool   seek_hold_ = false;
    int    seek_target_ = 0;
    QTimer seekHoldTimer_;
    void   highResSeek(int frame);     // send_manual_high_res_slider ＋ 表示保持
    void   clearSeekHold();
    int  FrameNo = 0;
    int  slider_No = 0;
    int  range_start_ = 0;
    int  range_end_ = 0;
    // 範囲つまみで再生位置を押し出したか（Widgets 版 RangeSlider の m_playChanged 相当）
    bool range_pushed_play_ = false;
    int  pushed_frame_ = 0;
    void pushPlayIntoRange();
    QString play_time_ = "00:00";
    // 開いたファイルの映像・音声の概要（動画情報に出す。canUseGpuDecode で読む）
    QString srcPixel_;          // "10bit 4:2:0 · PQ（表示 8bit）" など
    QString srcAudio_;          // "AAC LC · 48 kHz" など
    int     srcAudioCh_ = 0;    // 元のチャンネル数

    int  encode_state = STATE_NOT_ENCODE;
    bool wasCanceled = false;

    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();
};

#endif // PLAYERCONTROLLER_H
