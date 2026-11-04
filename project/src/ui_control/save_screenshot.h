#ifndef SAVE_SCREENSHOT_H
#define SAVE_SCREENSHOT_H

#pragma once
#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QString>

namespace Ui {
class save_screenshot;
}

// 画像保存設定ウィンドウ
//   編集中の値は edit_ に持ち、OK で g_AppSettings へ反映する。
//   キャンセル / Esc / × で閉じた場合は破棄 (次に開くと g_AppSettings の値に戻る)。
class save_screenshot : public QWidget
{
    Q_OBJECT

public:
    explicit save_screenshot(QWidget *parent = nullptr);
    ~save_screenshot();

protected:
    void showEvent(QShowEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    // 編集中の画像保存設定 (g_AppSettings の screenshot 関連の写し)
    struct Edit {
        QString folder;
        QString format;
        QString tiffCompression;
        int     jpegQuality = 90;
        QString jpegSubsampling;
        QString mode;
    };

    Ui::save_screenshot *ui;
    Edit edit_;

    void screenshot_path_dialog();
    void load_from_global();          // g_AppSettings → edit_ → UI (知らない値は既定値へ)
    void commit_to_global();          // edit_ → g_AppSettings
    // 選択中のフォーマットに応じて行を出し入れし、行とボタンを詰める
    void update_option_visibility();
};

#endif // SAVE_SCREENSHOT_H
