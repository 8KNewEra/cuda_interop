#include "src/ui_control/save_screenshot.h"
#include "info/logfile_control.h"
#include "qpainter.h"
#include "QStandardPaths"
#include "ui_save_screenshot.h"

save_screenshot::save_screenshot(QWidget *parent)
    : QWidget(parent),
    ui(new Ui::save_screenshot)
{
    ui->setupUi(this);

    ui->label_savescreenshotpath->setText(g_AppSettings.saveScreenshot_folder);
    QObject::connect(ui->pushButton_savescreenshotpath, &QPushButton::clicked,this, &save_screenshot::screenshot_path_dialog,Qt::QueuedConnection);

    ui->comboBox_savescreenshot_format->addItem("JPG (.jpg)");
    ui->comboBox_savescreenshot_format->addItem("PNG (.png)");
    ui->comboBox_savescreenshot_format->addItem("TIFF (.tiff)");
    ui->comboBox_savescreenshot_format->addItem("BMP (.bmp)");

    ui->comboBox_savescreenshot_pngmode->addItem("fast");
    ui->comboBox_savescreenshot_pngmode->addItem("normal");
    ui->comboBox_savescreenshot_pngmode->addItem("high");



    QObject::connect(ui->comboBox_savescreenshot_format, &QComboBox::currentIndexChanged,
                     this, [&](int index) {
                        if(index == 0){
                            g_AppSettings.screenshot_format = "jpg";
                        }else if(index == 1){
                            g_AppSettings.screenshot_format = "png";
                        }else if(index == 2){
                            g_AppSettings.screenshot_format = "tiff";
                        }else if(index == 3){
                            g_AppSettings.screenshot_format = "bmp";
                        }
        }, Qt::QueuedConnection);
    QObject::connect(ui->comboBox_savescreenshot_pngmode, &QComboBox::currentIndexChanged,
                     this, [&](int index) {
                        if(index == 0){
                            g_AppSettings.screenshot_mode = "fast";
                        }else if(index == 1){
                            g_AppSettings.screenshot_mode = "normal";
                        }else if(index == 2){
                            g_AppSettings.screenshot_mode = "high";
                        }
        }, Qt::QueuedConnection);

    auto emitIndexChanged = [&](QComboBox* box){
        int idx = box->currentIndex();
        Q_EMIT box->currentIndexChanged(idx);
    };

    {
        int index{};
        if(g_AppSettings.screenshot_format == "jpg"){
            index = 0;
        }else if(g_AppSettings.screenshot_format == "png"){
            index = 1;
        }else if(g_AppSettings.screenshot_format == "tiff"){
            index = 2;
        }else if(g_AppSettings.screenshot_format == "bmp"){
            index = 3;
        }
        ui->comboBox_savescreenshot_format->setCurrentIndex(index);
        emitIndexChanged(ui->comboBox_savescreenshot_format);
    }

    {
        int index{};
        if(g_AppSettings.screenshot_mode == "fast"){
            index = 0;
        }else if(g_AppSettings.screenshot_mode == "normal"){
            index = 1;
        }else if(g_AppSettings.screenshot_mode == "high"){
            index = 2;
        }
        ui->comboBox_savescreenshot_pngmode->setCurrentIndex(index);
        emitIndexChanged(ui->comboBox_savescreenshot_pngmode);
    }

    if (g_AppSettings.saveScreenshot_folder.isEmpty() || !QDir(g_AppSettings.saveScreenshot_folder).exists())
    {
        g_AppSettings.saveScreenshot_folder = QStandardPaths::writableLocation(
            QStandardPaths::PicturesLocation
            );

        if (g_AppSettings.saveScreenshot_folder.isEmpty() || !QDir(g_AppSettings.saveScreenshot_folder).exists())
        {
            g_AppSettings.saveScreenshot_folder = QDir::homePath();
        }
    }
}

save_screenshot::~save_screenshot()
{
    delete ui;
}

void save_screenshot::screenshot_path_dialog(){

    QString startDir = g_AppSettings.saveScreenshot_folder;

    if (startDir.isEmpty() || !QDir(startDir).exists())
    {
        startDir = QStandardPaths::writableLocation(
            QStandardPaths::PicturesLocation
            );

        if (startDir.isEmpty() || !QDir(startDir).exists())
        {
            startDir = QDir::homePath();
        }
    }

    QFileDialog* dlg = new QFileDialog(this);

    dlg->setAttribute(Qt::WA_DeleteOnClose);

    dlg->setWindowTitle(tr("フォルダーを選択"));

    dlg->setDirectory(startDir);

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

            // 必要なら設定に保存
            g_AppSettings.saveScreenshot_folder = folderPath;
            ui->label_savescreenshotpath->setText(g_AppSettings.saveScreenshot_folder);
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
