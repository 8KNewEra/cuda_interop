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
#include <deque>
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
class DecodeSettings;

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
    // 性能 HUD（Ctrl+Shift+P）。perf は 0.25 秒ごとの集計（段ごとの avg / max、フレーム間隔の列など）
    Q_PROPERTY(bool perfVisible READ perfVisible WRITE setPerfVisible NOTIFY displayChanged)
    Q_PROPERTY(QVariantMap perf READ perf NOTIFY perfChanged)

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
    // 表示（ウィンドウの色バッファ）の色深度。main がシーングラフの初期化後に渡す（10 / 8）
    void setDisplayBits(int bits) { displayBits_ = bits; emit infoChanged(); }
    // 画面に出るまでの深さ（DisplayProbe が渡す。displayprobe.h）
    //   モニター出力（NVIDIA の 8bpc / 10bpc。0 = 取れない）
    void setOutputBits(int bits)      { if (bits != outputBits_) { outputBits_ = bits; emit infoChanged(); } }
    //   画面合成の形式（8 / 10 / 16 = FP16。0 = 取れない）
    void setCompositionBits(int bits) { if (bits != compBits_)   { compBits_ = bits;   emit infoChanged(); } }
    //   画面合成を通るか（1 = 通る / 0 = 全画面で直接出している / -1 = 調べている・不明）
    void setComposed(int state)       { if (state != composed_)  { composed_ = state;  emit infoChanged(); } }
    //   NVIDIA の Deep color（OpenGL の 10bit ウィンドウを合成の外で出す）が有効か
    void setDeepColorScanout(bool on) { if (on != deepColor_)  { deepColor_ = on;    emit infoChanged(); } }

    // エンコード設定（QML には "encoder" として公開する）
    QObject* gaussianKernel() const;
    QObject* averagingKernel() const;
    QObject* mediaInfo() const;
    Q_INVOKABLE void showProperties();     // 開いているファイルの詳細情報を読んで画面を出す
    EncodeController* encoder() const { return encoder_; }
    ScreenshotSettings* screenshotSettings() const { return shotSettings_; }
    DecodeSettings* decodeSettings() const { return decodeSettings_; }   // マルチ GPU デコード（context property "decodeSettings"）

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
    bool perfVisible() const { return g_AppSettings.perf_hud; }
    void setPerfVisible(bool on);
    QVariantMap perf() const { return perf_; }

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

    QVariantMap  videoInfo() const;   // 作り置き（infoChanged / stateChanged ごとに 1 回だけ作る）
    QVariantList gpus() const;        // 同上
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
    Q_INVOKABLE void openDecodeSettings();   // ファイル → デコード設定…
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
    // 動画情報の推移グラフの記録の刻み（1 秒ごと。infoChanged 2 回に 1 回。InfoPanel.sample）
    void infoSampleTick();
    void histogramChanged();
    void perfChanged();
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
    QMetaObject::Connection encodeFinishedConn_;   // start_encode の完了の接続（エラーで終わったら外す）
    void start_decode_thread(const QString& filePath);
    void stop_decode_thread();
    bool canUseGpuDecode(const QString& filename);
    void decode_view(VideoFrame Frame, bool pause, bool reverse);
    void showFrame(const VideoFrame& f);   // 表示するコマの番号・時刻を画面へ（frameChanged）

    // ---- 表示のペース（vblank に合わせて出す。docs/DECODE_CORE.md 3.6）----
    //   安定した順方向の通常再生だけ、届いたコマをすぐ upload せず待ち行列に置き、実際の vblank（DWM）の格子に
    //   合わせた時刻に 1 枚ずつ出す。受け取りの返事（receve_decode_flag）は出したときに 1 枚ずつ返す
    //   （待ち行列のコマは「GUI の未受け取り ≦ 2」の枠の中。リング 6 段の勘定は変わらない）
    std::deque<VideoFrame> presentQueue_;
    QTimer  presentTimer_;              // 次のコマを出す時刻（Qt::PreciseTimer、単発）
    qint64  nextIdealNs_ = 0;           // 次のコマの理想の表示時刻（0 = 起点を取り直す）
    qint64  pacedFrameDurNs_ = 0;       // 1 コマの長さ（fps × 再生速度から）
    int     pacingEnv_ = -1;            // CUDA_VIEWER_PACING（0 で切る）。-1 = 未読
    int     pendingWaits_ = 0;          // 前のコマの swap 待ちで延ばした回数
    qint64  presentWakePlanNs_ = 0;     // 先頭のコマの upload を始める予定の時刻（表示の計測。タイマーの遅れ）
    qint64  presentTargetVNs_ = 0;      // 先頭のコマを出す vblank の時刻（schedulePresent）
    double  presentPerNs_ = 0;          // そのときの vblank の周期
    qint64  lastPresentedVNs_ = 0;      // 最後にコマを出した vblank（同じ vblank に 2 枚出さない）

    // ---- UI の定期更新を表示のペースの空き時間に寄せる（docs/DECODE_CORE.md 3.6）----
    //   性能 HUD（250ms）・動画情報（500ms）・推移の記録（1 秒）の通知は QML の評価と追加の描画を起こす。
    //   表示のペースの最中は印だけ付け、presentTick の swap の直後（次の upload まで 20ms 以上空いている）にまとめて出す
    enum UiTick { UiPerf = 1, UiInfo = 2 };
    int     pendingUi_ = 0;
    int     infoTicks_ = 0;
    bool    pacingActive() const { return !presentQueue_.empty() || presentTimer_.isActive(); }
    void    deferUi(int what);          // 表示のペースの最中なら印だけ、そうでなければすぐ出す
    void    emitUi(int what);
    void    flushDeferredUi();          // 印の付いた通知を出す（presentTick の swap の直後・待ち行列を空けたとき）

    // videoInfo / gpus の作り置き（バインディングごとに作り直すと 1 回の通知で数 ms かかっていた）
    QVariantMap  buildVideoInfo() const;
    QVariantList buildGpus() const;
    quint64              infoGen_ = 0;                 // infoChanged / stateChanged ごとに進む
    mutable quint64      videoInfoGen_ = ~0ull, gpusGen_ = ~0ull;
    mutable QVariantMap  videoInfoCache_;
    mutable QVariantList gpusCache_;
    bool    updatePacing();             // このコマを待ち行列に通すか（条件と 1 コマの長さの更新）
    void    schedulePresent();          // 先頭のコマを出す時刻にタイマーを掛ける
    void    presentTick();              // 先頭のコマを出す
    void    flushPresentQueue(bool showNewest);   // 待ち行列を空ける（showNewest なら最後のコマを出す）
    void    replyDecode();              // デコードへ 1 枚受け取ったと返す
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
    DecodeSettings* decodeSettings_ = nullptr;     // デコード設定（マルチ GPU デコード）
    // デコード設定を反映するために開き直したとき、開き終わったらこのフレームへ戻す（-1 = なし）
    int reopen_frame_ = -1;
    void reopenForDecodeSettings();

    decode_thread* decodestream = nullptr;
    QThread*       decode__thread = nullptr;
    QThread*       fps_view_thread = nullptr;
    fps_thread*    fpsstream = nullptr;
    info_thread*   infostream = nullptr;
    QTimer         infoTimer_;
    QTimer         perfTimer_;      // 性能 HUD の集計を取り出す（ON のときだけ動く）
    QVariantMap    perf_;
    void applyPerf();               // HUD の ON/OFF を計測とタイマーへ反映

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
    QString srcPixel_;          // "10bit 4:2:0 · PQ" など
    int     displayBits_ = 0;   // ウィンドウの色バッファの深さ（setDisplayBits。0 = まだ分からない）
    int     outputBits_  = 0;   // モニターへの出力の深さ（setOutputBits。0 = 取れない）
    int     compBits_    = 0;   // 画面合成の深さ（setCompositionBits。0 = 取れない）
    int     composed_    = 1;   // 画面合成を通るか（setComposed）
    bool    deepColor_   = false; // NVIDIA Deep color が有効か（setDeepColorScanout）
    QString srcAudio_;          // "AAC LC · 48 kHz" など
    int     srcAudioCh_ = 0;    // 元のチャンネル数

    int  encode_state = STATE_NOT_ENCODE;
    bool wasCanceled = false;

    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();
};

#endif // PLAYERCONTROLLER_H
