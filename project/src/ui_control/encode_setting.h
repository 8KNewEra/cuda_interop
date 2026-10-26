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
#include <QMessageBox>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include "src/main/__global__.h"

namespace Ui {
class encode_setting;
}

class encode_setting : public QWidget
{
    Q_OBJECT

public:
    explicit encode_setting(QWidget *parent = nullptr);
    ~encode_setting();
    void progress_bar(int value);
    void slider(int min,int max);
    void encode_end(QString encode_time);

signals:
    void signal_encode_start();
    void signal_encode_finished();
    void signal_encode_stop();

private:
    void closeEvent(QCloseEvent *event);
    void resizeEvent(QResizeEvent *event) override;
    Ui::encode_setting *ui;

    // -------------------------------------------------------------------------
    // ★ 可変の項目数を吸収する入れ物
    //
    //   .ui は固定座標なので、レート制御で行が消えるとその下が丸ごと空く。
    //   上半分（ファイル名〜GPU表）をスクロール領域へ入れ、行数に応じて
    //   詳細設定グループの高さと、その下の位置を計算し直す。
    //   ボタンと進捗バーだけはスクロールさせたくないので外に残し、
    //   常にウィンドウ下端へ貼り付ける。
    //
    //   ★ ウィンドウの大きさは項目数で変えないこと。
    //     行が消えるたびに窓が伸び縮みすると、設定を見比べている最中に
    //     位置が動いて落ち着かない。余った高さは下端に残して余白として
    //     見せ、入りきらない場合だけ縦スクロールで吸収する。
    // -------------------------------------------------------------------------
    // ラベルは日本語、ffmpeg のフラグはツールチップへ回す
    void setupLabels();

    void setupScrollArea();
    void layoutFooter();
    void relayoutContent();

    // ★ いちばん項目が多い状態の中身の高さ。
    //   これ以上ウィンドウを縦に伸ばしても余白が増えるだけなので、
    //   最大高さの上限に使う（モードを切り替えても動かないよう、
    //   いまのモードではなく最大のモードで計算する）。
    int  maxContentHeight() const;

    QScrollArea* scrollArea_ = nullptr;
    QWidget*     content_    = nullptr;

    // ★ いま何行見えているか。applyRcVisibility() が決め、
    //   relayoutContent() がそれを元に高さを組み直す。
    //   resizeEvent からも呼べるように状態として持っている。
    int  rate_rows_   = 1;       // ビットレート/品質スライダーの行数
    bool use_basis_   = true;    // 「ビットレート基準」の行を出しているか
    bool info_target_ = true;    // 換算行（目標）を出しているか
    bool info_max_    = false;   // 換算行（最大）を出しているか
    EncodeSettings& encodeSettings = EncodeSettingsManager::getInstance().getSettingsNonConst();
    const DecodeInfo& VideoInfo = DecodeInfoManager::getInstance().getSettings();

    // -------------------------------------------------------------------------
    // ★ 出力コンテナ
    //   muxer は拡張子から決まるので、コンボとパスは常に一致させておく。
    //   どちらを触っても他方が追従する（path_sync_ は相互呼び出しの門番）。
    // -------------------------------------------------------------------------
    void applyContainerToPath();      // コンボ -> パスの拡張子
    void syncContainerFromPath();     // パスの拡張子 -> コンボ
    void applyContainerCodecLimit();  // MOV では AV1 を出さない

    // ★ 音声コーデックはコンテナで入らない組み合わせがあるので、
    //   コンテナが変わるたびに項目を作り直す。
    //     MP4 : PCM 不可（ipcm タグになり対応プレーヤーが少ない）
    //     MOV : FLAC 不可（muxer が弾く）
    void rebuildAudioCodecs();
    // ★ 可逆（PCM / FLAC）ではビットレートの行ごと消す。
    //   レート制御の行と同じ扱いで、無効化ではなく非表示にする。
    void applyAudioVisibility();
    int  acodec_key_ = -1;             // 作り直しの要否判定
    int  audio_h_ = 128;               // 行数から決まる音声グループの高さ
    bool path_sync_ = false;

    void iniFile_UI_Control();
    int foundIndex(QString key,const QString& item);
    QString file_check(const QString &filePath);

    struct SettingEntry {
        QString codec;
        QString splitencode_items;
        int B_frame_items;
        QString preset_items;
        QString profile_items;
        QString rc_items;
        QString pass_items;
        int gop_items;
        int encode_tile;
    };

    QMap<int, SettingEntry> settingmap;
    std::vector<int> combo_index= {0,0,0,0,0,0,0,0,0,0};

    // ★ スライダーが示す値（Mbps）。単位は bitrate_basis に従うので、
    //   "total" のときはファイル全体の合計を指す。
    //   EncodeSettings へ入れるときに bitrate_unit() で 1 本あたりへ割り戻す。
    int target_bit_rate=100;
    int max_bit_rate=1000;
    bool encode_flag=false;

    // ビットレートの表示単位まわり
    // ★ スライダーと入力欄を揃える。本流は常にスライダー側なので、
    //   範囲や吸着を通った結果をこちらへ書き戻す。
    static void syncSpin(QSlider* slider, QSpinBox* spin);

    int  bitrate_unit() const;      // "total" ならタイル数、"stream" なら 1
    void applyBitrateRange();       // 基準とタイル数からスライダーの範囲を決める
    void updateBitrateInfo();       // 値ラベルと換算の表示を更新する

    // ★ レート制御で使わない行は「無効化」ではなく「非表示」にする。
    //   .ui が固定座標なので、見えている行を上から詰め直すところまで含む。
    void applyRcVisibility();

    // ★ コンボの行の出し入れ。view は内部で引く（null のとき何もしない）。
    //   QComboBox::view() は QAbstractItemView* なので、qobject_cast が
    //   失敗しうる。素で ->setRowHidden() を呼ぶと落ちる。
    static void hideRow(QComboBox* box, int row, bool hide);
    void combo_index_control(QComboBox* comboBox, int ini_index,int target_index, bool flag,bool index_change_flag);
    void fps_cmbobox_control();
    void combo_index_control2();
    void tile_index_control();
    void tile_split_exchange();
    void setGPUTable();

    // ★ GPU 表を行数ぶんの高さに詰める。
    //   外側のスクロール領域の中に表自身のスクロールが入れ子になると、
    //   ホイールが表に吸われて設定画面が動かなくなる。
    //   GPU が 2 枚のときに下へ余白が出るのも同時に防げる。
    void fitTableHeight();
    bool isAv1Supported(const GPUInfo& gpu);
    void applyAv1GpuRestriction();
    void restoreGpuRestriction();
    void updateGpuTileAllocation();
    std::vector<int> buildTileGpuMap();
    void multiGPUtabel_UI_Control(bool flag);
};

#endif // ENCODE_SETTING_H
