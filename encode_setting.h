#ifndef ENCODE_SETTING_H
#define ENCODE_SETTING_H

#include <QWidget>
#include <QPushButton>
#include <QFile>
#include "qfileinfo.h"
#include <QTextStream>
#include <QDebug>
#include <QFileDialog>
#include <QListView>
#include <QComboBox>
#include "__global__.h"

namespace Ui {
class encode_setting;
}

extern EncodeSettings g_encode_settings;

class encode_setting : public QWidget
{
    Q_OBJECT

public:
    explicit encode_setting(QWidget *parent = nullptr);
    ~encode_setting();
    void progress_bar(int value);
    void slider(int min,int max);
    void encode_end();

signals:
    void signal_encode_start();
    void signal_encode_finished();
    void signal_encode_stop();

private:
    void closeEvent(QCloseEvent *event);
    Ui::encode_setting *ui;
    EncodeSettings& settings = EncodeSettingsManager::getInstance().getSettingsNonConst();
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();
    void read_txt();
    void init_txt();
    void write_txt();
    int foundIndex(QString key,const QString& item);
    void extracted(QListView *&view, int &target_index, bool &flag);
    void combo_index_control(QComboBox* comboBox,QListView *view, int ini_index,int target_index, bool flag,bool index_change_flag);
    void combo_index_control2();
    QString file_check(const QString &filePath);
    struct SettingEntry {
        QString codec;
        int framerate_items;
        QString splitencode_items;
        int B_frame_items;
        QString preset_items;
        QString profile_items;
        QString rc_items;
        QString pass_items;
        int gop_items;
    };

    QMap<int, SettingEntry> settingmap;
    std::vector<int> combo_index= {0, 6, 0, 0, 3, 0, 0, 0,0,0};
    bool allow_overwrite;

    int target_bit_rate=100;
    int max_bit_rate=200;

    bool encode_flag=false;
};

#endif // ENCODE_SETTING_H
