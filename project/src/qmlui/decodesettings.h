#ifndef DECODESETTINGS_H
#define DECODESETTINGS_H

#include <QObject>
#include <QString>
#include <QVariantList>
#include <map>

// デコード設定（ファイル → デコード設定…。docs/MULTI_GPU_DECODE.md）
//   GPU ごとの重み（0..10）と転送の圧縮を編集し、apply() で g_GPUInfo / g_AppSettings へ反映する。
//   反映は次に開いたとき（apply(true) は今のファイルを開き直す）
class DecodeSettings : public QObject
{
    Q_OBJECT
    // [{id, name, sm, weight, canDecode, primary, dec, tiles}]（tiles = 今のファイルで受け持つ本数の見込み、-1 = なし）
    Q_PROPERTY(QVariantList gpus     READ gpus     NOTIFY changed)
    Q_PROPERTY(int     fileTiles READ fileTiles NOTIFY changed)   // 0 = GPU デコードのタイル動画を開いていない
    Q_PROPERTY(QString fileCodec READ fileCodec NOTIFY changed)
    Q_PROPERTY(bool    dirty     READ dirty     NOTIFY changed)
    // 編集中の重みで 2 枚以上の GPU を使うか（1 枚なら転送が無いので圧縮の設定は使わない）
    Q_PROPERTY(bool    multiGpu  READ multiGpu  NOTIFY changed)
    // 転送の圧縮: "none" / "cascaded" / "bitcomp" / "ans"、差分
    Q_PROPERTY(QString xferCodec READ xferCodec NOTIFY changed)
    Q_PROPERTY(bool    xferDelta READ xferDelta NOTIFY changed)
    Q_PROPERTY(QVariantList xferCodecOptions READ xferCodecOptions CONSTANT)

public:
    explicit DecodeSettings(QObject* parent = nullptr);

    QVariantList gpus() const;
    int     fileTiles() const { return fileTiles_; }
    QString fileCodec() const { return fileCodec_; }
    bool    dirty() const;
    bool    multiGpu() const;
    QString xferCodec() const { return xferCodec_; }
    bool    xferDelta() const { return xferDelta_; }
    QVariantList xferCodecOptions() const;

    // 開くたびに現在の設定から作り直す。tiles / codec は今のファイル（GPU デコードでなければ 0 / ""）
    void open(int tiles, const QString& codec);

    Q_INVOKABLE void setWeight(int deviceId, int weight);
    Q_INVOKABLE void setXferCodec(const QString& v);
    Q_INVOKABLE void setXferDelta(bool on);
    Q_INVOKABLE void apply(bool reopen);   // OK（reopen = 今のファイルを開き直して反映）
    Q_INVOKABLE void cancel();             // キャンセル / Esc

signals:
    void changed();
    void opened();
    void closed();
    void reopenRequested();   // apply(true)。PlayerController が今の位置から開き直す

private:
    std::map<int, int> weights_;   // deviceID → 編集中の重み
    int     fileTiles_ = 0;
    QString xferCodec_ = "none";
    bool    xferDelta_ = true;
    QString fileCodec_;
};

#endif // DECODESETTINGS_H
