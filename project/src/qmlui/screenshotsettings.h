#ifndef SCREENSHOTSETTINGS_H
#define SCREENSHOTSETTINGS_H

#include <QObject>
#include <QString>
#include <QVariantList>

// 画像保存設定（QML 版。Widgets 版 save_screenshot と同じ設定項目）
//   編集中の値はこのクラスが持ち、apply() で g_AppSettings へ反映する
//   cancel() / Esc は破棄（次に開くと g_AppSettings の値に戻る）
class ScreenshotSettings : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString folder          READ folder          NOTIFY changed)
    Q_PROPERTY(QString format          READ format          NOTIFY changed)
    Q_PROPERTY(QString tiffCompression READ tiffCompression NOTIFY changed)
    Q_PROPERTY(int     jpegQuality     READ jpegQuality     NOTIFY changed)
    Q_PROPERTY(QString jpegSubsampling READ jpegSubsampling NOTIFY changed)
    Q_PROPERTY(QString mode            READ mode            NOTIFY changed)

    // 行の出し入れ（Widgets 版 update_option_visibility と同じ条件）
    Q_PROPERTY(bool showTiffCompression READ showTiffCompression NOTIFY changed)
    Q_PROPERTY(bool showJpeg            READ showJpeg            NOTIFY changed)
    Q_PROPERTY(bool showMode            READ showMode            NOTIFY changed)

    // 選択肢 [{text, value}]（Select.qml にそのまま渡す）
    Q_PROPERTY(QVariantList formatOptions          READ formatOptions          CONSTANT)
    Q_PROPERTY(QVariantList tiffCompressionOptions READ tiffCompressionOptions CONSTANT)
    Q_PROPERTY(QVariantList jpegSubsamplingOptions READ jpegSubsamplingOptions CONSTANT)
    Q_PROPERTY(QVariantList modeOptions            READ modeOptions            CONSTANT)

public:
    explicit ScreenshotSettings(QObject* parent = nullptr);

    QString folder() const          { return folder_; }
    QString format() const          { return format_; }
    QString tiffCompression() const { return tiffCompression_; }
    int     jpegQuality() const     { return jpegQuality_; }
    QString jpegSubsampling() const { return jpegSubsampling_; }
    QString mode() const            { return mode_; }

    bool showTiffCompression() const { return format_ == "tiff"; }
    bool showJpeg() const            { return format_ == "jpg"; }
    bool showMode() const            { return format_ == "png" || (format_ == "tiff" && tiffCompression_ == "deflate"); }

    QVariantList formatOptions() const;
    QVariantList tiffCompressionOptions() const;
    QVariantList jpegSubsamplingOptions() const;
    QVariantList modeOptions() const;

    // 開くたびに現在の設定から作り直す（前回キャンセルした編集は残さない）
    void open();

    Q_INVOKABLE void setValue(const QString& key, const QVariant& value);
    Q_INVOKABLE void setJpegQuality(int value);
    Q_INVOKABLE void pickFolder();        // フォルダー選択ダイアログ
    Q_INVOKABLE void apply();             // OK
    Q_INVOKABLE void cancel();            // キャンセル / Esc

signals:
    void changed();
    void opened();
    void closed();

private:
    QString folder_;
    QString format_;
    QString tiffCompression_;
    int     jpegQuality_ = 90;
    QString jpegSubsampling_;
    QString mode_;

    void loadFromGlobal();     // g_AppSettings → 編集中の値（知らない値は既定値へ）
    void commitToGlobal();     // 編集中の値 → g_AppSettings
};

#endif // SCREENSHOTSETTINGS_H
