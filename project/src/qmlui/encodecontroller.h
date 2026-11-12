#ifndef ENCODECONTROLLER_H
#define ENCODECONTROLLER_H

// ==============================================================
// EncodeController（Qt Quick 版のエンコード設定）
//
//   Widgets 版 encode_setting のロジックを UI から切り離したもの。
//   値の持ち方・制約・換算・GPU 割り当ての規則は encode_setting と同じ。
//   違い:
//     ・コンボの「行を隠す」代わりに、選べる候補だけのリストを渡す
//       （QML 側は value で選択を持つのでインデックスのずれを気にしなくてよい）
//     ・制約の連鎖（タイル → コーデック → B フレーム／チューニング …）は normalize() で一括して解く
//
//   QML へは context property "encoder" として公開する。
// ==============================================================

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QElapsedTimer>
#include <QTimer>
#include <vector>
#include <algorithm>
#include <cmath>
#include "src/main/__global__.h"

class EncodeController : public QObject
{
    Q_OBJECT

    // ---- 状態 ----
    Q_PROPERTY(bool encoding  READ encoding  NOTIFY stateChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY stateChanged)
    Q_PROPERTY(bool finishedOk READ finishedOk NOTIFY stateChanged)
    Q_PROPERTY(int  progressValue READ progressValue NOTIFY progressChanged)
    Q_PROPERTY(int  progressMax   READ progressMax   NOTIFY progressChanged)
    Q_PROPERTY(double encodeFps   READ encodeFps     NOTIFY progressChanged)   // エンコード速度（直近 0.5 秒）
    Q_PROPERTY(QString summary READ summary NOTIFY changed)

    // ---- ファイル ----
    Q_PROPERTY(QString path          READ path          NOTIFY changed)
    Q_PROPERTY(bool    separateFiles READ separateFiles WRITE setSeparateFiles NOTIFY changed)

    // ---- 選択肢（[{text, value}]、選べるものだけ）と現在値 ----
    Q_PROPERTY(QVariantList containerOptions READ containerOptions NOTIFY changed)
    Q_PROPERTY(QVariantList codecOptions     READ codecOptions     NOTIFY changed)
    Q_PROPERTY(QVariantList framerateOptions READ framerateOptions NOTIFY changed)
    Q_PROPERTY(QVariantList splitOptions     READ splitOptions     NOTIFY changed)
    Q_PROPERTY(QVariantList tileOptions      READ tileOptions      NOTIFY changed)
    Q_PROPERTY(QVariantList presetOptions    READ presetOptions    NOTIFY changed)
    Q_PROPERTY(QVariantList tuneOptions      READ tuneOptions      NOTIFY changed)
    Q_PROPERTY(QVariantList gopOptions       READ gopOptions       NOTIFY changed)
    Q_PROPERTY(QVariantList bframeOptions    READ bframeOptions    NOTIFY changed)
    Q_PROPERTY(QVariantList rcOptions        READ rcOptions        NOTIFY changed)
    Q_PROPERTY(QVariantList passOptions      READ passOptions      NOTIFY changed)
    Q_PROPERTY(QVariantList basisOptions     READ basisOptions     NOTIFY changed)
    Q_PROPERTY(QVariantList audioCodecOptions READ audioCodecOptions NOTIFY changed)

    Q_PROPERTY(QVariant container READ container NOTIFY changed)
    Q_PROPERTY(QVariant codec     READ codec     NOTIFY changed)
    Q_PROPERTY(QVariant framerate READ framerate NOTIFY changed)
    Q_PROPERTY(QVariant split     READ split     NOTIFY changed)
    Q_PROPERTY(QVariant tile      READ tile      NOTIFY changed)
    Q_PROPERTY(QVariant preset    READ preset    NOTIFY changed)
    Q_PROPERTY(QVariant tune      READ tune      NOTIFY changed)
    Q_PROPERTY(QVariant gop       READ gop       NOTIFY changed)
    Q_PROPERTY(QVariant bframes   READ bframes   NOTIFY changed)
    Q_PROPERTY(QVariant rc        READ rc        NOTIFY changed)
    Q_PROPERTY(QVariant pass      READ pass      NOTIFY changed)
    Q_PROPERTY(QVariant basis     READ basis     NOTIFY changed)
    Q_PROPERTY(QVariant audioCodec READ audioCodec NOTIFY changed)
    Q_PROPERTY(bool bframeEditable READ bframeEditable NOTIFY changed)

    // ---- ビットレート / 品質 ----
    Q_PROPERTY(bool showTarget  READ showTarget  NOTIFY changed)
    Q_PROPERTY(bool showMax     READ showMax     NOTIFY changed)
    Q_PROPERTY(bool showQuality READ showQuality NOTIFY changed)
    Q_PROPERTY(bool showBasis   READ showBasis   NOTIFY changed)
    Q_PROPERTY(int  targetRate  READ targetRate  NOTIFY changed)
    Q_PROPERTY(int  targetMin   READ rateMin     NOTIFY changed)
    Q_PROPERTY(int  targetMax   READ targetHi    NOTIFY changed)
    Q_PROPERTY(int  maxRate     READ maxRate     NOTIFY changed)
    Q_PROPERTY(int  maxMin      READ rateMin     NOTIFY changed)
    Q_PROPERTY(int  maxMax      READ rateHi      NOTIFY changed)
    Q_PROPERTY(int  quality     READ quality     NOTIFY changed)
    Q_PROPERTY(QString qualityLabel READ qualityLabel NOTIFY changed)
    Q_PROPERTY(QString qualityFlag  READ qualityFlag  NOTIFY changed)
    Q_PROPERTY(QString rateUnit     READ rateUnit     NOTIFY changed)
    Q_PROPERTY(QString rateInfoTarget READ rateInfoTarget NOTIFY changed)
    Q_PROPERTY(QString rateInfoMax    READ rateInfoMax    NOTIFY changed)
    Q_PROPERTY(QString vramInfo       READ vramInfo       NOTIFY changed)

    // ---- 音声 ----
    Q_PROPERTY(bool audioAvailable READ audioAvailable NOTIFY changed)
    Q_PROPERTY(bool audioEnabled   READ audioEnabled   WRITE setAudioEnabled NOTIFY changed)
    Q_PROPERTY(bool audioLossless  READ audioLossless  NOTIFY changed)
    Q_PROPERTY(int  audioBitrate   READ audioBitrate   NOTIFY changed)   // kbps
    Q_PROPERTY(int  audioGain      READ audioGain      NOTIFY changed)   // dB

    // ---- マルチ GPU ----
    Q_PROPERTY(QVariantList gpus  READ gpus  NOTIFY changed)
    // 使用率・メモリは別プロパティで 1 秒ごとに更新（gpus の行を作り直さないので操作中の重みに影響しない）
    //   gpus と同じ並び。[{id, encUsage, gpuUsage, g3d, cuda, memUsed, memMax}]（g3d / cuda は取れない環境で -1）
    Q_PROPERTY(QVariantList gpuStats READ gpuStats NOTIFY statsChanged)
    Q_PROPERTY(bool multiGpu      READ multiGpu NOTIFY changed)

public:
    explicit EncodeController(QObject* parent = nullptr);

    // ---- PlayerController から ----
    // 設定画面を開くとき（Widgets 版 encode_setting::slider 相当）。進捗の範囲もここで決める
    void prepare(int progressMax);
    void setProgress(int value);
    // 最後まで終わった（Widgets 版 encode_end 相当）
    void encodeEnded(const QString& elapsed);
    // エンコーダの初期化に失敗した（開始前の状態へ戻す）
    void encodeFailed(const QString& message);

    // ---- QML から ----
    Q_INVOKABLE void browsePath();
    Q_INVOKABLE void setValue(const QString& key, const QVariant& value);
    Q_INVOKABLE void setTargetRate(int v);
    Q_INVOKABLE void setMaxRate(int v);
    Q_INVOKABLE void setQuality(int v);
    Q_INVOKABLE void setAudioBitrate(int kbps);
    Q_INVOKABLE void setAudioGain(int db);
    Q_INVOKABLE void setGpuWeight(int deviceId, int weight);
    Q_INVOKABLE void start();
    Q_INVOKABLE void cancel();
    // 閉じる。エンコード中は閉じない（false を返す）
    Q_INVOKABLE bool close();

    // ---- getter ----
    bool encoding() const { return encoding_; }
    QString statusText() const { return status_; }
    bool finishedOk() const { return finished_ok_; }
    int  progressValue() const { return progress_; }
    double encodeFps() const { return encode_fps_; }
    int  progressMax() const { return progress_max_; }
    QString summary() const;

    QString path() const { return es().encode_path; }
    bool separateFiles() const { return es().separate_files; }
    void setSeparateFiles(bool on);

    QVariantList containerOptions() const;
    QVariantList codecOptions() const;
    QVariantList framerateOptions() const;
    QVariantList splitOptions() const;
    QVariantList tileOptions() const;
    QVariantList presetOptions() const;
    QVariantList tuneOptions() const;
    QVariantList gopOptions() const;
    QVariantList bframeOptions() const;
    QVariantList rcOptions() const;
    QVariantList passOptions() const;
    QVariantList basisOptions() const;
    QVariantList audioCodecOptions() const;

    QVariant container() const { return es().container; }
    QVariant codec() const     { return es().codec; }
    QVariant framerate() const { return es().save_fps; }
    QVariant split() const     { return es().split_encode_mode; }
    QVariant tile() const      { return es().encode_tile; }
    QVariant preset() const    { return es().preset; }
    QVariant tune() const      { return es().tune; }
    QVariant gop() const       { return es().gop_size; }
    QVariant bframes() const   { return es().b_frames; }
    QVariant rc() const        { return es().rc_mode; }
    QVariant pass() const      { return es().pass_mode; }
    QVariant basis() const     { return es().bitrate_basis; }
    QVariant audioCodec() const { return es().audio_codec; }
    bool bframeEditable() const { return bframe_capable_; }

    bool showTarget() const;
    bool showMax() const;
    bool showQuality() const;
    bool showBasis() const { return showTarget() || showMax(); }
    int  targetRate() const { return target_; }
    int  maxRate() const { return max_; }
    int  rateMin() const { return unit(); }
    int  rateHi() const { return 1000 * unit(); }
    int  targetHi() const { return es().rc_mode == "vbr" ? max_ : rateHi(); }
    int  quality() const { return es().cq; }
    QString qualityLabel() const;
    QString qualityFlag() const;
    QString rateUnit() const;
    QString rateInfoTarget() const;
    QString rateInfoMax() const;
    QString vramInfo() const;

    bool audioAvailable() const;
    bool audioEnabled() const { return es().audio_enable; }
    void setAudioEnabled(bool on);
    bool audioLossless() const;
    int  audioBitrate() const { return std::clamp(es().audio_bitrate / 1000, 64, 512); }
    int  audioGain() const { return std::clamp(int(std::lround(es().audio_gain_db)), -30, 30); }

    QVariantList gpus() const;
    QVariantList gpuStats() const;
    bool multiGpu() const { return g_GPUInfo.size() > 1 && !encoding_; }

signals:
    void changed();
    void stateChanged();
    void progressChanged();
    void statsChanged();

    // PlayerController へ（Widgets 版 encode_setting の signal と同じ意味）
    void encodeStartRequested();
    void encodeStopRequested();
    void closed();

private:
    // 書き込みは EncodeSettingsManager の friend としてだけ行う
    EncodeSettings& es() { return EncodeSettingsManager::getInstance().getSettingsNonConst(); }
    const EncodeSettings& es() const { return EncodeSettingsManager::getInstance().getSettings(); }
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    void loadFromSettings();       // ini → 表示値（Widgets 版 iniFile_UI_Control 相当）
    void normalize();              // 選べない値を選べる値へ寄せる（制約の連鎖をまとめて解く）
    void applyRateRange();         // Widgets 版 applyBitrateRange 相当
    void applyContainerToPath();
    void syncContainerFromPath();
    void updateTileAllocation();   // Widgets 版 updateGpuTileAllocation 相当
    void applyAv1Restriction();    // Widgets 版 apply/restoreGpuRestriction 相当
    std::vector<int> buildTileGpuMap() const;
    static QString fileCheck(const QString& filePath);

    int  unit() const { return es().bitrate_basis == "total" ? std::max(1, es().encode_tile) : 1; }
    bool primaryAv1() const;
    int  canvasW() const { return VideoInfo.width  * VideoInfo.width_scale; }
    int  canvasH() const { return VideoInfo.height * VideoInfo.height_scale; }
    bool tileAllowed(int tiles) const;
    bool codecAllowed(const QString& c) const;
    bool bframeAllowed(int bf) const;
    bool tuneAllowed(const QString& t) const;
    static void tileSplit(int tiles, int& wt, int& ht);

    // 表示中のビットレート（Mbps）。"total" のときはファイル全体の合計
    //   EncodeSettings へは start() で 1 本あたりに割り戻して入れる
    int target_ = 100;
    int max_    = 1000;

    bool bframe_capable_ = true;   // CC 7.5 以上（Turing 以降）
    bool encoding_ = false;
    bool canceled_ = false;
    bool finished_ok_ = false;
    QString status_;
    int  progress_ = 0;
    QTimer statsTimer_;               // 設定画面を開いている間だけ動く（1 秒）
    // エンコード速度の計測
    double        encode_fps_ = 0.0;
    QElapsedTimer fpsTimer_;          // 区間計測（0.5 秒ごとに更新）
    int           fpsBase_ = 0;       // 区間の開始フレーム数
    QElapsedTimer totalTimer_;        // 開始からの経過（平均 fps 用）
    int  progress_max_ = 1;

    // GPU ごとの割り当てタイル数（deviceID → tiles）
    std::vector<std::pair<int,int>> gpu_tiles_;
};

#endif // ENCODECONTROLLER_H
