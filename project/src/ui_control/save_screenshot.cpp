#include "src/ui_control/save_screenshot.h"
#include "info/logfile_control.h"
#include "qpainter.h"
#include "QStandardPaths"
#include "ui_save_screenshot.h"
#include <QKeyEvent>
#include <QSignalBlocker>

// コンボボックスの並び順と設定値の対応
//   (表示名, 設定値) の組で持つと、並び替えや追加で index がずれない
static const struct { const char* label; const char* value; } kFormats[] = {
    {"JPG (.jpg)",   "jpg"},
    {"PNG (.png)",   "png"},
    {"TIFF (.tiff)", "tiff"},
    {"BMP (.bmp)",   "bmp"},
};
static const struct { const char* label; const char* value; } kTiffCompressions[] = {
    {"可逆 (Deflate)", "deflate"},
    {"非圧縮",         "none"},
};
// 4:4:4 = 色差を間引かない (文字・細線・色の境界がにじまない、ファイルは大きい)
// 4:2:0 = 縦横とも半分 (一般的な写真・動画と同じ、最も小さい)
static const struct { const char* label; const char* value; } kJpegSubsamplings[] = {
    {"4:4:4 (高画質)", "444"},
    {"4:2:2",          "422"},
    {"4:2:0 (標準)",   "420"},
};
static const struct { const char* label; const char* value; } kModes[] = {
    {"fast",   "fast"},
    {"normal", "normal"},
    {"high",   "high"},
};

// value に対応する index。ini に知らない値が入っていたら fallback (AppSettings の既定値) の index
//   (設定値の正規化はこのクラスが担当する。logfile_control は読み書きだけ)
template <size_t N, class T>
static int indexOf(const T (&table)[N], const QString& value, const QString& fallback)
{
    for (size_t i = 0; i < N; ++i)
        if (value == table[i].value) return (int)i;
    for (size_t i = 0; i < N; ++i)
        if (fallback == table[i].value) return (int)i;
    return 0;
}

// 保存先フォルダが無効なら 画像フォルダ → ホーム の順に補う
static QString validFolder(const QString& folder)
{
    if (!folder.isEmpty() && QDir(folder).exists()) return folder;
    const QString pictures = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    if (!pictures.isEmpty() && QDir(pictures).exists()) return pictures;
    return QDir::homePath();
}

save_screenshot::save_screenshot(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::save_screenshot)
{
    ui->setupUi(this);

    for (const auto& f : kFormats)          ui->comboBox_savescreenshot_format->addItem(f.label);
    for (const auto& c : kTiffCompressions) ui->comboBox_savescreenshot_tiffcompression->addItem(QString::fromUtf8(c.label));
    for (const auto& j : kJpegSubsamplings) ui->comboBox_savescreenshot_jpegsubsampling->addItem(QString::fromUtf8(j.label));
    for (const auto& m : kModes)            ui->comboBox_savescreenshot_pngmode->addItem(m.label);

    // ---- UI の操作は edit_ だけを書き換える (OK を押すまで g_AppSettings は変えない)
    QObject::connect(ui->pushButton_savescreenshotpath, &QPushButton::clicked,this, &save_screenshot::screenshot_path_dialog,Qt::QueuedConnection);
    QObject::connect(ui->comboBox_savescreenshot_format, &QComboBox::currentIndexChanged,
                     this, [this](int index) {
                         if (index < 0) return;
                         edit_.format = kFormats[index].value;
                         update_option_visibility();
                     });
    QObject::connect(ui->comboBox_savescreenshot_tiffcompression, &QComboBox::currentIndexChanged,
                     this, [this](int index) {
                         if (index < 0) return;
                         edit_.tiffCompression = kTiffCompressions[index].value;
                         update_option_visibility();
                     });
    QObject::connect(ui->horizontalSlider_jpegquality, &QSlider::valueChanged,
                     this, [this](int value) {
                         edit_.jpegQuality = value;
                         ui->label_jpegquality_value->setText(QString::number(value));
                     });
    QObject::connect(ui->comboBox_savescreenshot_jpegsubsampling, &QComboBox::currentIndexChanged,
                     this, [this](int index) {
                         if (index < 0) return;
                         edit_.jpegSubsampling = kJpegSubsamplings[index].value;
                     });
    QObject::connect(ui->comboBox_savescreenshot_pngmode, &QComboBox::currentIndexChanged,
                     this, [this](int index) {
                         if (index < 0) return;
                         edit_.mode = kModes[index].value;
                     });

    QObject::connect(ui->pushButton_ok, &QPushButton::clicked, this, [this]() {
        commit_to_global();
        close();
    });
    QObject::connect(ui->pushButton_cancel, &QPushButton::clicked, this, &QWidget::close);

    // 起動時 (ini 読み込み直後) に一度だけ正規化した値を g_AppSettings へ戻す。
    // ini に知らない値があっても、保存処理側は常に正しい値だけを見ればよい。
    load_from_global();
    commit_to_global();
}

save_screenshot::~save_screenshot()
{
    delete ui;
}

// 開くたびに現在の設定から作り直す (前回キャンセルした編集は残さない)
void save_screenshot::showEvent(QShowEvent *event)
{
    load_from_global();
    QWidget::showEvent(event);
}

// QWidget は QDialog と違い Esc で閉じないので自前で扱う。Enter は OK
void save_screenshot::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Escape) {
        close();
        return;
    }
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        commit_to_global();
        close();
        return;
    }
    QWidget::keyPressEvent(event);
}

void save_screenshot::load_from_global()
{
    const AppSettings def;                  // 既定値は __global__.h の 1 か所だけで決める

    edit_.folder          = validFolder(g_AppSettings.saveScreenshot_folder);
    edit_.format          = kFormats[indexOf(kFormats, g_AppSettings.screenshot_format, def.screenshot_format)].value;
    edit_.tiffCompression = kTiffCompressions[indexOf(kTiffCompressions, g_AppSettings.screenshot_tiff_compression, def.screenshot_tiff_compression)].value;
    edit_.jpegQuality     = qBound(0, g_AppSettings.screenshot_jpeg_quality, 100);
    edit_.jpegSubsampling = kJpegSubsamplings[indexOf(kJpegSubsamplings, g_AppSettings.screenshot_jpeg_subsampling, def.screenshot_jpeg_subsampling)].value;
    edit_.mode            = kModes[indexOf(kModes, g_AppSettings.screenshot_mode, def.screenshot_mode)].value;

    // UI へ反映 (ここでの setCurrentIndex 等で signal が飛ばないよう止める)
    {
        const QSignalBlocker b1(ui->comboBox_savescreenshot_format);
        const QSignalBlocker b2(ui->comboBox_savescreenshot_tiffcompression);
        const QSignalBlocker b3(ui->horizontalSlider_jpegquality);
        const QSignalBlocker b4(ui->comboBox_savescreenshot_jpegsubsampling);
        const QSignalBlocker b5(ui->comboBox_savescreenshot_pngmode);
        ui->label_savescreenshotpath->setText(edit_.folder);
        ui->comboBox_savescreenshot_format->setCurrentIndex(indexOf(kFormats, edit_.format, edit_.format));
        ui->comboBox_savescreenshot_tiffcompression->setCurrentIndex(indexOf(kTiffCompressions, edit_.tiffCompression, edit_.tiffCompression));
        ui->horizontalSlider_jpegquality->setValue(edit_.jpegQuality);
        ui->label_jpegquality_value->setText(QString::number(edit_.jpegQuality));
        ui->comboBox_savescreenshot_jpegsubsampling->setCurrentIndex(indexOf(kJpegSubsamplings, edit_.jpegSubsampling, edit_.jpegSubsampling));
        ui->comboBox_savescreenshot_pngmode->setCurrentIndex(indexOf(kModes, edit_.mode, edit_.mode));
    }
    update_option_visibility();
}

void save_screenshot::commit_to_global()
{
    g_AppSettings.saveScreenshot_folder       = edit_.folder;
    g_AppSettings.screenshot_format           = edit_.format;
    g_AppSettings.screenshot_tiff_compression = edit_.tiffCompression;
    g_AppSettings.screenshot_jpeg_quality     = edit_.jpegQuality;
    g_AppSettings.screenshot_jpeg_subsampling = edit_.jpegSubsampling;
    g_AppSettings.screenshot_mode             = edit_.mode;
}

// TIFF 圧縮は TIFF のときだけ、JPEG 品質 / 色差サンプリングは JPEG のときだけ、
// 圧縮レベルは PNG と TIFF(可逆) のときだけ表示する。
// .ui は座標指定なので、見えている行を上から詰め直し、その下に OK / キャンセルを置いて
// ウィンドウの高さも合わせる。
void save_screenshot::update_option_visibility()
{
    const bool tiff    = edit_.format == "tiff";
    const bool png     = edit_.format == "png";
    const bool jpg     = edit_.format == "jpg";
    const bool tiffZip = tiff && edit_.tiffCompression == "deflate";

    // 1 行 = 左の見出し + 右側の部品 (最大 2 つ)
    struct Row { QWidget* label; QWidget* widgets[2]; bool visible; };
    const Row rows[] = {
        {ui->label_2,               {ui->comboBox_savescreenshot_format, nullptr},          true},
        {ui->label_tiffcompression, {ui->comboBox_savescreenshot_tiffcompression, nullptr}, tiff},
        {ui->label_jpegquality,     {ui->horizontalSlider_jpegquality, ui->label_jpegquality_value}, jpg},
        {ui->label_jpegsubsampling, {ui->comboBox_savescreenshot_jpegsubsampling, nullptr},  jpg},
        {ui->label_3,               {ui->comboBox_savescreenshot_pngmode, nullptr},         png || tiffZip},
    };

    const int kFirstRowY = 33, kRowPitch = 24, kWindowWidth = 379;
    int y = kFirstRowY;
    for (const Row& r : rows) {
        r.label->setVisible(r.visible);
        for (QWidget* w : r.widgets) if (w) w->setVisible(r.visible);
        if (!r.visible) continue;
        r.label->move(r.label->x(), y);
        for (QWidget* w : r.widgets) if (w) w->move(w->x(), w == r.widgets[0] ? y + 1 : y);
        y += kRowPitch;
    }

    // OK / キャンセル (右寄せ、Windows の並び: OK が左)
    const int kButtonGap = 6, kBottomMargin = 9;
    const int by = y + kButtonGap;
    ui->pushButton_ok->move(ui->pushButton_ok->x(), by);
    ui->pushButton_cancel->move(ui->pushButton_cancel->x(), by);
    // 幅は .ui の設計値 (379) で、高さは表示中の行数で固定 (ユーザーによるリサイズ不可)
    setFixedSize(kWindowWidth, by + ui->pushButton_ok->height() + kBottomMargin);
}

void save_screenshot::screenshot_path_dialog(){

    QFileDialog* dlg = new QFileDialog(this);

    dlg->setAttribute(Qt::WA_DeleteOnClose);

    dlg->setWindowTitle(tr("フォルダーを選択"));

    dlg->setDirectory(validFolder(edit_.folder));

    // フォルダー選択
    dlg->setFileMode(QFileDialog::Directory);

    // 「フォルダーを選択」だけにする
    dlg->setOption(
        QFileDialog::ShowDirsOnly,
        true
        );

    connect(
        dlg,
        &QFileDialog::fileSelected,
        this,
        [this](const QString& folderPath)
        {
            qDebug()
            << "選択されたフォルダー:"
            << folderPath;

            // OK を押すまでは編集中の値だけ変える
            edit_.folder = folderPath;
            ui->label_savescreenshotpath->setText(edit_.folder);
        }
        );

    connect(
        dlg,
        &QFileDialog::rejected,
        this,
        []
        {
            qDebug()
            << "フォルダー選択がキャンセルされました";
        }
        );

    // 非同期表示
    dlg->open();
}
