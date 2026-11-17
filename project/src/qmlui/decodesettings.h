#ifndef DECODESETTINGS_H
#define DECODESETTINGS_H

#include <QObject>
#include <QString>
#include <QVariantList>
#include <map>

// マルチ GPU デコードの設定（QML 版。ファイル → デコード設定…）
//   GPU ごとの「デコードの重み」（0..10）を編集する。タイルの割り振りは decode_gpu_map.h（assignDecodeTiles）
//   編集中の値はこのクラスが持ち、apply() で g_GPUInfo（decode_weight）へ反映する。cancel() / Esc は破棄
//   デコーダはファイルを開くときに作るので、反映は次に開いたとき（apply(true) なら今のファイルを開き直す）
class DecodeSettings : public QObject
{
    Q_OBJECT
    // [{id, name, sm, weight, canDecode, primary, dec, tiles}]（tiles = 今のファイルで受け持つ本数の見込み）
    Q_PROPERTY(QVariantList gpus     READ gpus     NOTIFY changed)
    // 今のファイルの情報（GPU デコードで開いているとき）。タイル数 0 = 見込みを出さない
    Q_PROPERTY(int     fileTiles READ fileTiles NOTIFY changed)
    Q_PROPERTY(QString fileCodec READ fileCodec NOTIFY changed)
    // 今の設定から変わっているか（「開き直して反映」を出すかどうか）
    Q_PROPERTY(bool    dirty     READ dirty     NOTIFY changed)
    // 転送の圧縮（試験的。nvCOMP）: "none" / "cascaded" / "bitcomp" / "ans"、差分の前処理
    Q_PROPERTY(QString xferCodec READ xferCodec NOTIFY changed)
    Q_PROPERTY(bool    xferDelta READ xferDelta NOTIFY changed)
    Q_PROPERTY(QVariantList xferCodecOptions READ xferCodecOptions CONSTANT)

public:
    explicit DecodeSettings(QObject* parent = nullptr);

    QVariantList gpus() const;
    int     fileTiles() const { return fileTiles_; }
    QString fileCodec() const { return fileCodec_; }
    bool    dirty() const;
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
    void reopenRequested();   // apply(true) のとき。PlayerController が今の位置から開き直す

private:
    std::map<int, int> weights_;   // deviceID → 重み（編集中）
    int     fileTiles_ = 0;
    QString xferCodec_ = "none";
    bool    xferDelta_ = true;
    QString fileCodec_;
};

#endif // DECODESETTINGS_H
