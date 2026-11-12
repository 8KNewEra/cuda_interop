#ifndef MEDIAPROBE_H
#define MEDIAPROBE_H

#include <QObject>
#include <QString>
#include <QVariantList>
#include <QFutureWatcher>

// 動画ファイルの詳細情報（プロパティ画面）
//   FFmpeg（libavformat）でコンテナとストリームの情報を読み、
//   「全般 / 映像 #n / 音声 #n / …」のセクションごとに 項目名・値 の表にする
//   読み取りは QtConcurrent の別スレッド（大きなファイルでも画面を止めない）
//   PlayerController::showProperties() → open() → 完了で opened()（QML がダイアログを出す）
class MediaProbe : public QObject
{
    Q_OBJECT
    // [{title, rows: [{k, v}]}]
    Q_PROPERTY(QVariantList sections READ sections NOTIFY changed)
    Q_PROPERTY(bool    busy     READ busy     NOTIFY changed)
    Q_PROPERTY(QString error    READ error    NOTIFY changed)
    Q_PROPERTY(QString fileName READ fileName NOTIFY changed)

public:
    explicit MediaProbe(QObject* parent = nullptr);
    ~MediaProbe() override;

    QVariantList sections() const { return sections_; }
    bool    busy() const     { return busy_; }
    QString error() const    { return error_; }
    QString fileName() const { return fileName_; }

    Q_INVOKABLE void open(const QString& path);   // 読み取り開始（終わると changed）
    Q_INVOKABLE void copyToClipboard() const;
    Q_INVOKABLE void saveAs() const;               // テキストファイルに保存（保存先を選ぶ）

    // 表をテキストにする（コピー・保存用。MediaInfo のような「項目 : 値」の並び）
    QString toText() const;

signals:
    void changed();
    void opened();     // 画面を出す合図

private:
    struct Result { QVariantList sections; QString error; };
    static Result probe(const QString& path);

    QVariantList sections_;
    bool    busy_ = false;
    QString error_;
    QString fileName_;
    QFutureWatcher<Result> watcher_;
};

#endif // MEDIAPROBE_H
