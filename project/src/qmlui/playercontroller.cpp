#include "src/qmlui/playercontroller.h"
#include "src/qmlui/videoengine.h"

#include "src/videoprocess/avidecode.h"
#include "src/videoprocess/cpudecode.h"
#include "src/videoprocess/decode_thread.h"
#include "src/videoprocess/nvgpudecode.h"
#include "src/main/fps_thread.h"
#include "src/info/info_thread.h"
#include "src/info/logfile_control.h"
#include "src/ui_control/encode_setting.h"
#include "src/ui_control/save_screenshot.h"

#include <QCoreApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <algorithm>

extern "C" {
#include <libavformat/avformat.h>
}

PlayerController::PlayerController(QObject* parent)
    : QObject(parent)
{
    // ★ ini は GPU 情報の突き合わせより先に読む（MainWindow と同じ順序）
    logFile = new logfile_control();

    // 高精度シークで目標のコマが来なかった場合（範囲外への指定など）の保険
    seekHoldTimer_.setSingleShot(true);
    seekHoldTimer_.setInterval(1500);
    connect(&seekHoldTimer_, &QTimer::timeout, this, [this] { clearSeekHold(); });

    infoTimer_.setInterval(500);
    connect(&infoTimer_, &QTimer::timeout, this, [this] {
        if (g_AppSettings.videoInfo_flag) emit infoChanged();
    });
}

PlayerController::~PlayerController()
{
    stop_decode_thread();
    stop_info_thread();
    infoTimer_.stop();

    delete encodeSetting;  encodeSetting = nullptr;
    delete saveScreenshot; saveScreenshot = nullptr;
    delete engine_;        engine_ = nullptr;

    // ★ 最後に ini を保存（logfile_control のデストラクタが書く）
    delete logFile;        logFile = nullptr;
}

bool PlayerController::initialize()
{
    engine_ = new VideoEngine();
    if (!engine_->initialize()) {
        qWarning() << "[PlayerController] VideoEngine の初期化に失敗しました";
        return false;
    }

    // ★ Queued にしておく（encode_mode() は start_encode() の途中から呼ばれる）
    connect(engine_, &VideoEngine::encode_error, this,
            &PlayerController::encode_error_handler, Qt::QueuedConnection);
    connect(engine_, &VideoEngine::histogramChanged, this, &PlayerController::histogramChanged);

    start_info_thread();
    infoTimer_.start();

    // エンコード設定（当面は既存の Widgets ウィンドウ）
    encodeSetting = new encode_setting();
    encodeSetting->setWindowModality(Qt::ApplicationModal);
    encodeSetting->hide();
    connect(encodeSetting, &encode_setting::signal_encode_start,
            this, &PlayerController::start_encode, Qt::QueuedConnection);
    connect(encodeSetting, &encode_setting::signal_encode_finished,
            this, &PlayerController::finished_encode, Qt::QueuedConnection);

    // スクリーンショット保存先（同上）
    saveScreenshot = new save_screenshot();
    saveScreenshot->setWindowModality(Qt::ApplicationModal);
    saveScreenshot->hide();

    ready_ = true;
    emit stateChanged();
    return true;
}

// ==============================================================
// 表示・フィルタ・再生設定
// ==============================================================
void PlayerController::setInfoVisible(bool on)
{
    if (g_AppSettings.videoInfo_flag == on) return;
    g_AppSettings.videoInfo_flag = on;
    emit displayChanged();
    emit infoChanged();
}

void PlayerController::setHistVisible(bool on)
{
    if (g_AppSettings.histgram_flag == on) return;
    g_AppSettings.histgram_flag = on;
    emit displayChanged();
}

void PlayerController::setGaussian(bool on)
{
    if (g_AppSettings.gaussianfilterEnabled == on) return;
    g_AppSettings.gaussianfilterEnabled = on;
    if (engine_) engine_->filter_change_flag = true;
    emit filterChanged();
}

void PlayerController::setSobel(bool on)
{
    if (g_AppSettings.sobelfilterEnabled == on) return;
    g_AppSettings.sobelfilterEnabled = on;
    if (engine_) engine_->filter_change_flag = true;
    emit filterChanged();
}

void PlayerController::setAveraging(bool on)
{
    if (g_AppSettings.averagingfilterEnabled == on) return;
    g_AppSettings.averagingfilterEnabled = on;
    if (engine_) engine_->filter_change_flag = true;
    emit filterChanged();
}

void PlayerController::allFiltersOff()
{
    g_AppSettings.gaussianfilterEnabled  = false;
    g_AppSettings.sobelfilterEnabled     = false;
    g_AppSettings.averagingfilterEnabled = false;
    if (engine_) engine_->filter_change_flag = true;
    emit filterChanged();
}

int PlayerController::jumpValue() const
{
    switch (g_AppSettings.frame_jump_mode) {
    case JUMP_MODE_FRAME:       return g_AppSettings.jumpValueFrame;
    case JUMP_MODE_TARGETFRAME: return g_AppSettings.jumpValueFrameNo;
    default:                    return g_AppSettings.jumpValueSecond;
    }
}

void PlayerController::setJumpMode(int mode)
{
    if (mode < JUMP_MODE_SECOND || mode > JUMP_MODE_TARGETFRAME) return;
    if (g_AppSettings.frame_jump_mode == mode) return;
    g_AppSettings.frame_jump_mode = mode;
    emit jumpChanged();
}

// MainWindow::get_jump_value と同じ範囲に丸める
void PlayerController::setJumpValue(int value)
{
    const bool open = run_decode_thread && VideoInfo.fps > 0;
    const int  maxF = VideoInfo.max_framesNo;

    switch (g_AppSettings.frame_jump_mode) {
    case JUMP_MODE_SECOND:
        if (value < 1) value = 1;
        if (open && value > maxF / VideoInfo.fps) value = int(maxF / VideoInfo.fps);
        g_AppSettings.jumpValueSecond = value;
        break;
    case JUMP_MODE_FRAME:
        if (value < 1) value = 1;
        if (open && value > maxF) value = maxF;
        g_AppSettings.jumpValueFrame = value;
        break;
    case JUMP_MODE_TARGETFRAME:
        if (value < 0) value = 0;
        if (open && value > maxF) value = maxF;
        g_AppSettings.jumpValueFrameNo = value;
        break;
    }
    emit jumpChanged();
}

QString PlayerController::jumpLabel(bool forward) const
{
    switch (g_AppSettings.frame_jump_mode) {
    case JUMP_MODE_FRAME:
        return QString("%1%2f").arg(forward ? "+" : "-").arg(g_AppSettings.jumpValueFrame);
    case JUMP_MODE_TARGETFRAME:
        return QString("#%1").arg(g_AppSettings.jumpValueFrameNo);
    default:
        return QString("%1%2s").arg(forward ? "+" : "-").arg(g_AppSettings.jumpValueSecond);
    }
}

void PlayerController::setSpeed(double ratio)
{
    if (ratio <= 0.0) return;
    g_AppSettings.video_speed_ratio = ratio;
    if (fpsstream) fpsstream->change_speed(g_AppSettings.video_speed_ratio * VideoInfo.fps);
    emit speedChanged();
}

void PlayerController::setVolume(int value)
{
    value = std::clamp(value, 0, 100);
    if (g_AppSettings.audio_volume == value) return;
    g_AppSettings.audio_volume = value;   // audio_decode が読む
    emit volumeChanged();
}

// ==============================================================
// 情報表示
// ==============================================================
int PlayerController::maxFrames() const
{
    return run_decode_thread ? VideoInfo.max_framesNo : 0;
}

QString PlayerController::formatFrames(int frames) const
{
    if (VideoInfo.fps <= 0.0) return "00:00";
    const double t = frames / VideoInfo.fps;
    const int h = int(t / 3600);
    const int m = (int(t) % 3600) / 60;
    const int s = int(std::fmod(t, 60.0));
    return (VideoInfo.max_hour > 0) ? QString::asprintf("%02d:%02d:%02d", h, m, s)
                                    : QString::asprintf("%02d:%02d", m, s);
}

QVariantMap PlayerController::videoInfo() const
{
    QVariantMap m;
    m["fileName"]      = QString::fromStdString(VideoInfo.Name);
    m["decoder"]       = QString::fromStdString(VideoInfo.Codec);
    m["decodeMode"]    = VideoInfo.decode_mode.trimmed();
    const QSize s      = engine_ ? engine_->displaySize() : QSize();
    m["width"]         = s.width();
    m["height"]        = s.height();
    m["fps"]           = VideoInfo.fps;
    m["maxFrame"]      = VideoInfo.max_framesNo;
    m["frameNo"]       = frameNo();
    m["hasAudio"]      = VideoInfo.audio;
    m["audioChannels"] = VideoInfo.audio_channels;
    m["renderFps"]     = engine_ ? engine_->fps : 0.0;
    m["bitDepth"]      = VideoInfo.bitdepth;
    return m;
}

QVariantList PlayerController::gpus() const
{
    QVariantList list;
    for (const GPUInfo& g : g_GPUInfo) {
        QVariantMap m;
        m["id"]       = g.deviceID;
        m["name"]     = g.deviceName;
        m["sm"]       = QString("sm_%1%2").arg(g.CC_major).arg(g.CC_minor);
        m["usage"]    = g.GPU_Usage;
        m["memUsed"]  = g.Memory_Usage;
        m["memMax"]   = g.Max_Memory_Usage;
        m["primary"]  = (g.deviceID == g_openglDeviceID);
        list << m;
    }
    return list;
}

QVariantMap PlayerController::histogram() const
{
    return engine_ ? engine_->histStats() : QVariantMap();
}

// ==============================================================
// ファイル
// ==============================================================
bool PlayerController::isSupportedFile(const QUrl& url) const
{
    const QString ext = QFileInfo(url.toLocalFile()).suffix().toLower();
    return ext == "mp4" || ext == "avi" || ext == "mkv" || ext == "mov";
}

void PlayerController::openFileDialog()
{
    QString startDir = QFileInfo(g_AppSettings.decode_path).absolutePath();
    if (startDir.isEmpty() || !QDir(startDir).exists())
        startDir = QDir::homePath();

    auto* dlg = new QFileDialog(nullptr);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(tr("動画ファイルを開く"));
    dlg->setDirectory(startDir);
    dlg->setFileMode(QFileDialog::ExistingFile);
    dlg->setNameFilter(tr("動画ファイル (*.mp4 *.avi *.mkv *.mov)"));
    connect(dlg, &QFileDialog::fileSelected, this, [this](const QString& filePath) {
        openFile(QUrl::fromLocalFile(filePath));
    });
    dlg->open();
}

void PlayerController::openFile(const QUrl& url)
{
    const QString filePath = url.isLocalFile() ? url.toLocalFile() : url.toString();
    if (!isSupportedFile(QUrl::fromLocalFile(filePath))) {
        QMessageBox::warning(nullptr, tr("非対応フォーマット"), tr("非対応フォーマットです"), QMessageBox::Ok);
        return;
    }
    qDebug() << "Open file:" << filePath;
    g_AppSettings.decode_path = filePath;
    closeFile();
    start_decode_thread(filePath);
}

void PlayerController::closeFile()
{
    if (run_decode_thread) stop_decode_thread();
    if (engine_) engine_->reset();
    FrameNo = 0;
    play_time_ = "00:00";
    range_start_ = range_end_ = 0;
    emit frameChanged();
    emit rangeChanged();
    emit stateChanged();
}

// GPU デコードできるか（MainWindow::canUseGpuDecode と同じ判定）
//   ★ 元の実装は H264/HEVC/AV1 の分岐で avformat_close_input を
//     呼ばずに return していたので、ここでは必ず閉じる。
bool PlayerController::canUseGpuDecode(const QString& filename)
{
    const QByteArray path = filename.toUtf8();
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.constData(), nullptr, nullptr) < 0)
        return false;

    bool ok = false;
    if (avformat_find_stream_info(fmt, nullptr) >= 0) {
        for (unsigned i = 0; i < fmt->nb_streams; i++) {
            const AVCodecParameters* par = fmt->streams[i]->codecpar;
            if (par->codec_type != AVMEDIA_TYPE_VIDEO) continue;
            if (par->codec_id == AV_CODEC_ID_H264)
                ok = (par->width <= 4096 && par->height <= 4096);
            else if (par->codec_id == AV_CODEC_ID_HEVC || par->codec_id == AV_CODEC_ID_AV1)
                ok = (par->width <= 8192 && par->height <= 8192);
            break;
        }
    }
    avformat_close_input(&fmt);
    return ok;
}

void PlayerController::start_decode_thread(const QString& filePath)
{
    if (run_decode_thread) return;

    if (canUseGpuDecode(filePath)) {
        decodestream = new nvgpudecode(filePath);
    } else {
        const QString ext = QFileInfo(filePath).suffix().toLower();
        if (ext == "mp4") {
            decodestream = new cpudecode(filePath);
        } else if (ext == "avi") {
            decodestream = new avidecode(filePath);
        } else {
            QMessageBox::warning(nullptr, tr("ファイルを開けません"), tr("非対応の動画です"), QMessageBox::Ok);
            return;
        }
    }

    decode__thread = new QThread;
    decodestream->moveToThread(decode__thread);

    connect(decodestream, &decode_thread::send_decode_image, this, &PlayerController::decode_view, Qt::QueuedConnection);
    connect(decodestream, &decode_thread::send_slider_info, this, &PlayerController::init_decodethread_complete);
    connect(decodestream, &decode_thread::heavy_process_signal, this, &PlayerController::setControlsEnabled);

    connect(this, &PlayerController::send_manual_resumeplayback, decodestream, &decode_thread::resumePlayback);
    connect(this, &PlayerController::send_manual_pause,          decodestream, &decode_thread::pausePlayback);
    connect(this, &PlayerController::send_manual_reverse,        decodestream, &decode_thread::reversePlayback);
    connect(this, &PlayerController::send_manual_back1frame,     decodestream, &decode_thread::back1frame, Qt::QueuedConnection);
    connect(this, &PlayerController::send_manual_go1frame,       decodestream, &decode_thread::go1frame, Qt::QueuedConnection);
    connect(this, &PlayerController::send_manual_slider,         decodestream, &decode_thread::sliderPlayback);
    connect(this, &PlayerController::send_manual_range_end_slider,   decodestream, &decode_thread::slider_range_end);
    connect(this, &PlayerController::send_manual_range_start_slider, decodestream, &decode_thread::slider_range_start);
    connect(this, &PlayerController::send_manual_high_res_slider,    decodestream, &decode_thread::high_res_sliderPlayback);

    connect(decodestream, &decode_thread::finished, decode__thread, &QThread::quit, Qt::SingleShotConnection);
    connect(decode__thread, &QThread::finished, decodestream, &QObject::deleteLater, Qt::SingleShotConnection);
    connect(decode__thread, &QThread::finished, decode__thread, &QObject::deleteLater, Qt::SingleShotConnection);

    connect(decodestream, &decode_thread::decode_error, this, [this](const QString& error) {
        closeFile();
        QMetaObject::invokeMethod(this, [error]() {
            QMessageBox::warning(nullptr, tr("デコードエラー"),
                                 tr("デコード中にエラーが発生しました:\n%1").arg(error), QMessageBox::Ok);
        }, Qt::QueuedConnection);
    }, Qt::SingleShotConnection);

    connect(decode__thread, &QThread::started, this, [this]() {
        run_decode_thread = true;
        emit stateChanged();
        QMetaObject::invokeMethod(decodestream, "startProcessing", Qt::QueuedConnection);
    }, Qt::SingleShotConnection);

    decode__thread->start();
}

void PlayerController::stop_decode_thread()
{
    if (!run_decode_thread) return;
    run_decode_thread = false;
    steady_play_frames_ = 0;
    render_active_ = false;
    play_intent_ = false;
    seekHoldTimer_.stop();
    seek_hold_ = false;
    setControlsEnabled(false);

    disconnect(decodestream, nullptr, this, nullptr);
    disconnect(this, nullptr, decodestream, nullptr);

    decodestream->stopProcessing();
    decode__thread->quit();
    decode__thread->wait();

    stop_fps_thread();

    // ★ 寿命は QThread::finished → deleteLater に任せている
    decodestream = nullptr;
    decode__thread = nullptr;
    emit stateChanged();
}

void PlayerController::init_decodethread_complete()
{
    setControlsEnabled(true);

    range_start_ = 0;
    range_end_   = VideoInfo.max_framesNo;
    emit rangeChanged();
    emit stateChanged();
    emit infoChanged();

    qDebug() << "Framerate:" << VideoInfo.fps << "MaxFrames:" << VideoInfo.max_framesNo;
    setPlayIntent(true);          // 開いた直後は再生から始まる
    start_fps_thread(VideoInfo.fps * g_AppSettings.video_speed_ratio);
}

void PlayerController::setControlsEnabled(bool on)
{
    if (controls_enabled_ == on) return;
    controls_enabled_ = on;
    emit stateChanged();
}

// ==============================================================
// 表示（MainWindow::decode_view と同じ手順）
// ==============================================================
void PlayerController::decode_view(VideoFrame Frame, bool pause, bool reverse)
{
    const bool playingBefore = playing();
    const bool reverseBefore = reverse_flag;
    pause_flag   = pause;
    reverse_flag = reverse;
    FrameNo      = Frame.FrameNo;

    // ★ 範囲つまみで押し出した後は、実際に表示されたコマを「確定位置」にする。
    //   押し出しは軽いシークなので、狙った番号（つまみの位置）とずれたコマが
    //   出ることがある。離したときに狙った番号へ高精度シークし直すと、
    //   表示中の再生位置が後から戻って見えるので、表示されたほうに合わせる。
    //   （範囲外のコマはシーク前の古いフレームなので採らない）
    if (range_pushed_play_ && Frame.FrameNo >= range_start_ && Frame.FrameNo <= range_end_)
        pushed_frame_ = Frame.FrameNo;

    // 高精度シークの目標に着いたら表示の保持を外す
    if (seek_hold_ && Frame.FrameNo == seek_target_)
        clearSeekHold();

    if (!run_decode_thread) return;

    // ★ 1 表示フレームにつき「1 本だけ」接続する（リング追い越し防止）
    connect(this, &PlayerController::decode_please, decodestream,
            &decode_thread::receve_decode_flag, Qt::SingleShotConnection);

    play_time_ = (VideoInfo.max_hour > 0)
                     ? QString::asprintf("%02d:%02d:%02d", Frame.hour, Frame.minute, Frame.second)
                     : QString::asprintf("%02d:%02d", Frame.minute, Frame.second);
    slider_No = Frame.FrameNo;
    emit frameChanged();
    // ★ コマ送りでも 1 枚だけ pause=false のフレームが来るので、
    //   数枚続いたときだけ「再生中」とみなす（止まるほうは即座に反映）
    //   さらに、ユーザーが再生を指示していない（シーク中など）ときは数えない。
    const bool activeBefore = render_active_;
    if (pause_flag || !play_intent_ || seek_hold_) {
        steady_play_frames_ = 0;
        render_active_ = false;
    } else if (steady_play_frames_ < kSteadyPlayFrames) {
        if (++steady_play_frames_ >= kSteadyPlayFrames) render_active_ = true;
    }

    if (playingBefore != playing() || reverseBefore != reverse_flag || activeBefore != render_active_)
        emit stateChanged();

    // OpenGL へ画像を渡して描画。一時停止中は再描画だけ（フィルタ変更・スクショ用）
    if (!pause_flag && encode_state != STATE_ENCODE_READY) {
        engine_->upload(Frame);
    } else if (encode_state == STATE_NOT_ENCODE) {
        engine_->rerender(Frame);
    }

    if (encode_state != STATE_ENCODE_READY)
        emit decode_please();
}

// ==============================================================
// 再生操作（MainWindow の各 *_pushbutton_control と同じ）
// ==============================================================
void PlayerController::togglePlay()
{
    if (!run_decode_thread) return;
    if (pause_flag || reverse_flag) { setPlayIntent(true);  emit send_manual_resumeplayback(); }
    else                            { setPlayIntent(false); emit send_manual_pause(); }
}

void PlayerController::stop()
{
    if (!run_decode_thread) return;
    setRangeStartFrame(0);
    setRangeEndFrame(VideoInfo.max_framesNo);
    range_pushed_play_ = false;
    setPlayIntent(false);
    clearSeekHold();
    emit send_manual_slider(0);
}

void PlayerController::reverse()
{
    if (!run_decode_thread) return;
    setPlayIntent(true);
    emit send_manual_reverse();
}

void PlayerController::stepBack()    { jumpBy(false); }
void PlayerController::stepForward() { jumpBy(true); }

void PlayerController::jumpBy(bool forward)
{
    if (!run_decode_thread) return;
    setPlayIntent(false);
    emit send_manual_pause();

    // ★ 起点は表示中の番号（シーク中なら目標）。連打しても前の目標から進む
    const int base = frameNo();

    switch (g_AppSettings.frame_jump_mode) {
    case JUMP_MODE_SECOND: {
        const int d = int(VideoInfo.fps * g_AppSettings.jumpValueSecond);
        highResSeek(forward ? base + d : base - d);
        break;
    }
    case JUMP_MODE_FRAME:
        if (g_AppSettings.jumpValueFrame == 1) {
            if (forward) emit send_manual_go1frame();
            else         emit send_manual_back1frame();
        } else {
            const int d = g_AppSettings.jumpValueFrame;
            highResSeek(forward ? base + d : base - d);
        }
        break;
    case JUMP_MODE_TARGETFRAME:
        highResSeek(g_AppSettings.jumpValueFrameNo);
        break;
    }
}

void PlayerController::toRangeStart()
{
    if (!run_decode_thread) return;
    setPlayIntent(false);
    emit send_manual_pause();
    highResSeek(VideoInfo.start_range_framesNo);
}

void PlayerController::toRangeEnd()
{
    if (!run_decode_thread) return;
    setPlayIntent(false);
    emit send_manual_pause();
    highResSeek(VideoInfo.end_range_framesNo);
}

void PlayerController::cutStart()
{
    if (!run_decode_thread) return;
    emit send_manual_pause();
    setPlayIntent(false);
    const int f = frameNo();
    setRangeStartFrame(f);
    range_pushed_play_ = false;
    highResSeek(f);
}

void PlayerController::cutEnd()
{
    if (!run_decode_thread) return;
    emit send_manual_pause();
    setPlayIntent(false);
    const int f = frameNo();
    setRangeEndFrame(f);
    range_pushed_play_ = false;
    highResSeek(f);
}

// ★ 再生位置は再生範囲の中に収める（旧 RangeSlider と同じ）
void PlayerController::scrub(int frame)
{
    if (!run_decode_thread) return;
    setPlayIntent(false);
    clearSeekHold();
    emit send_manual_slider(std::clamp(frame, range_start_, range_end_));
}

void PlayerController::scrubRelease(int frame)
{
    if (!run_decode_thread) return;
    setPlayIntent(false);
    highResSeek(std::clamp(frame, range_start_, range_end_));
}

// 範囲を狭めて再生位置が外に出たら、つまみ側へ押し出す。
//   ドラッグ中は軽いシーク（send_manual_slider）で追従させ、
//   離したときに rangeDragFinished() で高精度シークして確定する。
void PlayerController::pushPlayIntoRange()
{
    int target = FrameNo;
    if (range_pushed_play_) target = pushed_frame_;   // まだデコードが追いついていない
    const int clamped = std::clamp(target, range_start_, range_end_);
    if (clamped == target) return;

    range_pushed_play_ = true;
    pushed_frame_ = clamped;
    FrameNo = clamped;            // 表示は先に動かす（デコード結果で上書きされる）
    setPlayIntent(false);
    clearSeekHold();
    emit frameChanged();
    emit send_manual_slider(clamped);
}

void PlayerController::rangeDragFinished()
{
    if (!run_decode_thread || !range_pushed_play_) return;
    range_pushed_play_ = false;
    highResSeek(pushed_frame_);
}

void PlayerController::highResSeek(int frame)
{
    // 送る番号はそのまま（範囲外の扱いはデコーダ側の丸めに任せる）。
    // 表示はデコーダが丸める先＝範囲内に合わせておく。
    seek_target_ = std::clamp(frame, range_start_, range_end_);
    seek_hold_ = (seek_target_ != FrameNo);
    if (seek_hold_) seekHoldTimer_.start();
    emit frameChanged();
    emit send_manual_high_res_slider(frame);
}

void PlayerController::clearSeekHold()
{
    seekHoldTimer_.stop();
    if (!seek_hold_) return;
    seek_hold_ = false;
    emit frameChanged();
}

void PlayerController::setPlayIntent(bool on)
{
    if (play_intent_ == on) return;
    play_intent_ = on;
    if (!on && render_active_) {
        render_active_ = false;
        steady_play_frames_ = 0;
        emit stateChanged();
    }
}

void PlayerController::setRangeStartFrame(int frame)
{
    if (!run_decode_thread) return;
    frame = std::clamp(frame, 0, range_end_);
    range_start_ = frame;
    emit rangeChanged();
    emit send_manual_range_start_slider(frame);
    pushPlayIntoRange();
}

void PlayerController::setRangeEndFrame(int frame)
{
    if (!run_decode_thread) return;
    frame = std::clamp(frame, range_start_, VideoInfo.max_framesNo);
    range_end_ = frame;
    emit rangeChanged();
    emit send_manual_range_end_slider(frame);
    pushPlayIntoRange();
}

// ==============================================================
// スクリーンショット
// ==============================================================
void PlayerController::screenshot()
{
    if (!run_decode_thread || !engine_) return;
    engine_->screenshot_flag = true;
    setControlsEnabled(false);
    emit screenshotStateChanged(true);

    connect(engine_, &VideoEngine::screenshot_finished, this, [this]() {
        setControlsEnabled(true);
        emit screenshotStateChanged(false);
    }, Qt::SingleShotConnection);
}

void PlayerController::openScreenshotSettings()
{
    if (saveScreenshot) saveScreenshot->show();
}

// ==============================================================
// スレッド
// ==============================================================
void PlayerController::start_fps_thread(double target_fps)
{
    if (fpsstream) return;
    fpsstream = new fps_thread(target_fps);
    fps_view_thread = new QThread;
    fpsstream->moveToThread(fps_view_thread);
    connect(fpsstream, &fps_thread::fps_signal, decodestream, &decode_thread::processFrame, Qt::UniqueConnection);
    fpsstream->start();
}

// エンコード中はタイマー駆動を外し、ハンドシェイク駆動に一本化する
void PlayerController::set_fps_driving(bool on)
{
    if (!fpsstream || !decodestream) return;
    if (on)
        connect(fpsstream, &fps_thread::fps_signal, decodestream, &decode_thread::processFrame, Qt::UniqueConnection);
    else
        disconnect(fpsstream, &fps_thread::fps_signal, decodestream, &decode_thread::processFrame);
}

void PlayerController::stop_fps_thread()
{
    if (!fps_view_thread || !fpsstream) return;

    fpsstream->stop();
    fpsstream->wait();          // ★ run() のループが抜けるのを待つ
    fps_view_thread->quit();
    fps_view_thread->wait();

    delete fpsstream;
    delete fps_view_thread;
    fps_view_thread = nullptr;
    fpsstream = nullptr;
}

void PlayerController::start_info_thread()
{
    infostream = new info_thread;
    infostream->start();
}

void PlayerController::stop_info_thread()
{
    if (!infostream) return;
    infostream->quit();
    infostream->wait();
    delete infostream;
    infostream = nullptr;
}

// ==============================================================
// エンコード（MainWindow と同じ段取り）
// ==============================================================
void PlayerController::openEncodeSettings()
{
    if (!run_decode_thread || !encodeSetting) return;

    encode_state = STATE_ENCODE_READY;
    engine_->encode_mode(encode_state);
    decodestream->encode_state = encode_state;
    emit stateChanged();

    // ★ ここから先はタイマー駆動を切り離す（二重駆動の防止）
    set_fps_driving(false);
    setPlayIntent(false);
    emit send_manual_pause();

    encodeSetting->slider(0, VideoInfo.end_range_framesNo - VideoInfo.start_range_framesNo);
    encodeSetting->show();
}

void PlayerController::start_encode()
{
    if (!run_decode_thread) return;
    qDebug() << "エンコード開始";

    QElapsedTimer timer;
    timer.start();

    wasCanceled = false;
    connect(encodeSetting, &encode_setting::signal_encode_stop, this, [this]() {
        wasCanceled = true;
    }, Qt::SingleShotConnection);
    connect(decodestream, &decode_thread::decode_end, this, [this]() {
        wasCanceled = true;
    }, Qt::SingleShotConnection);
    connect(this, &PlayerController::decode_please, decodestream,
            &decode_thread::receve_decode_flag, Qt::SingleShotConnection);

    encode_state = STATE_ENCODING;
    set_fps_driving(false);
    decodestream->encode_state = encode_state;
    engine_->encode_mode(encode_state);
    engine_->MinFrame = VideoInfo.start_range_framesNo;
    engine_->MaxFrame = VideoInfo.end_range_framesNo;
    emit stateChanged();

    emit send_manual_high_res_slider(VideoInfo.start_range_framesNo);
    emit send_manual_resumeplayback();

    // ★ 起動キックは「再生状態にしてから」1 回だけ
    emit decode_please();

    connect(engine_, &VideoEngine::encode_finished, this, [this, timer]() {
        wasCanceled = true;
        emit send_manual_pause();

        encode_state = STATE_ENCODE_READY;
        if (decodestream) decodestream->encode_state = encode_state;
        engine_->encode_mode(encode_state);
        emit stateChanged();

        const double seconds = timer.nsecsElapsed() / 1e9;
        encodeSetting->encode_end(QString::number(seconds, 'f', 3) + "秒");
    }, Qt::SingleShotConnection);

    // 進捗の更新（MainWindow と同じくイベントを回しながら待つ）
    while (true) {
        encodeSetting->progress_bar(engine_->encode_FrameCount);
        if (wasCanceled) {
            engine_->MaxFrame = slider_No;
            break;
        }
        if (engine_->encode_FrameCount > (engine_->MaxFrame - engine_->MinFrame))
            break;
        QCoreApplication::processEvents();
    }
}

void PlayerController::encode_error_handler(const QString& error)
{
    wasCanceled = true;
    emit send_manual_pause();

    encode_state = STATE_ENCODE_READY;
    if (decodestream) decodestream->encode_state = encode_state;
    if (engine_) {
        engine_->encode_mode(encode_state);
        engine_->encode_FrameCount = 0;
    }
    emit stateChanged();

    qWarning() << "[PlayerController] encode error:" << error;
    QMessageBox::warning(nullptr, tr("エンコード開始エラー"),
                         tr("エンコーダの初期化に失敗しました。\n\n%1\n\n"
                            "解像度・分割数・B-frame 数を下げると改善する場合があります。").arg(error),
                         QMessageBox::Ok);
}

void PlayerController::finished_encode()
{
    encode_state = STATE_NOT_ENCODE;
    if (fpsstream) fpsstream->change_speed(g_AppSettings.video_speed_ratio * VideoInfo.fps);
    engine_->encode_mode(encode_state);
    if (decodestream) decodestream->encode_state = encode_state;
    emit stateChanged();

    if (!run_decode_thread) return;

    // ★ 通常再生に戻すのでタイマー駆動を復帰させる
    set_fps_driving(true);
    setPlayIntent(true);
    emit send_manual_resumeplayback();
    emit decode_please();
}
