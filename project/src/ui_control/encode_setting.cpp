#include "src/ui_control/encode_setting.h"
#include "qevent.h"
#include "qspinbox.h"
#include "ui_encode_setting.h"

encode_setting::encode_setting(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::encode_setting)
{
    ui->setupUi(this);

    settingmap[0] = {"h264_nvenc","0",0,"p1","default","cbr","1pass",1,1};
    settingmap[1] = {"hevc_nvenc","1",1,"p2","hq","vbr","2pass-quarter-res",15,2};
    settingmap[2] = {"av1_nvenc","",2,"p3","ll","cq","2pass-full-res",30,4};
    settingmap[3] = {"","",3,"p4","ull","","",60,8};
    settingmap[4] = {"","",4,"p5","lossless","","",120,0};
    settingmap[5] = {"","",5,"p6","","","",250,0};
    settingmap[6] = {"","",6,"p7","","","",30,0};
    settingmap[7] = {"","",7,"p8","","","",0,0};
    settingmap[8] = {"","",8,"","","","",0,0};
    settingmap[9] = {"","",9,"","","","",0,0};
    settingmap[10] = {"","",10,"","","","",0,0};

    //ファイルパス
    {
        QObject::connect(ui->filesave_pushButton, &QPushButton::clicked, this, [&]() {
            QString filter = "MP4 Files (*.mp4);;All Files (*.*)";
            QFileInfo fileInfo(encodeSettings.encode_path);

            QString fileName = QFileDialog::getSaveFileName(
                this,
                tr("名前を付けて保存"),
                fileInfo.absolutePath(),        // 初期ディレクトリ
                filter
                );

            if (!fileName.isEmpty()) {
                ui->label_filepath->setText(fileName);
                encodeSettings.encode_path = fileName;
            }
        }, Qt::QueuedConnection);
    }

    //コーデック
    {
        QStringList codec_items;
        //プライマリGPUがAV1エンコードに対応しているかを確認
        codec_items << "h264_nvenc" << "hevc_nvenc";
        for (const auto &gpu : g_GPUInfo)
        {
            if (gpu.openglEnable)
            {
                if (gpu.CC_major > 8 || (gpu.CC_major == 8 && gpu.CC_minor >= 9))
                    codec_items << "av1_nvenc";

                break;
            }
        }

        ui->comboBox_codec->addItems(codec_items);
        QObject::connect(ui->comboBox_codec, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.codec = settingmap[index].codec;
            combo_index_control2();
            setGPUTable();
        }, Qt::QueuedConnection);
    }

    //スプリットエンコード
    {
        QStringList splitencode_items;
        splitencode_items << "Disable" << "Enable" ;
        ui->comboBox_splitencode->addItems(splitencode_items);
        QObject::connect(ui->comboBox_splitencode, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.split_encode_mode = settingmap[index].splitencode_items;
        }, Qt::QueuedConnection);
    }

    //タイルエンコードプロファイル
    {
        QStringList tile_items;
        tile_items <<"1 Tile (通常)"<<"2 Tiles (2 × 1)"<<"4 Tiles (2 × 2)"<<"8 Tiles (4 × 2)";
        ui->comboBox_tile->addItems(tile_items);
        QObject::connect(ui->comboBox_tile, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.encode_tile = settingmap[index].encode_tile;
            tile_split_exchange();
            updateGpuTileAllocation();

            if(VideoInfo.width*VideoInfo.width_scale/encodeSettings.width_tile>4096||VideoInfo.height*VideoInfo.height_scale/encodeSettings.height_tile>4096){
                qobject_cast<QListView*>(ui->comboBox_codec->view())->setRowHidden(0,true);
                if(ui->comboBox_codec->currentIndex()==0){
                    ui->comboBox_codec->setCurrentIndex(1);
                }
            }else{
                qobject_cast<QListView*>(ui->comboBox_codec->view())->setRowHidden(0,false);
            }

            combo_index_control2();
        }, Qt::QueuedConnection);
    }

    //Bフレーム
    {
        QStringList B_frame_items;
        if(g_GPUInfo[0].CC_major > 7 || (g_GPUInfo[0].CC_major == 7 && g_GPUInfo[0].CC_minor >= 5)){
            B_frame_items << "0" << "1"<<"2"<<"3"<<"4"<<"5"<<"6"<<"7";
            ui->comboBox_b_frame->addItems(B_frame_items);
            QObject::connect(ui->comboBox_b_frame, &QComboBox::currentIndexChanged, this, [&](int index) {
                encodeSettings.b_frames = settingmap[index].B_frame_items;
            }, Qt::QueuedConnection);
        }else{
            B_frame_items << "0" ;
            ui->comboBox_b_frame->addItems(B_frame_items);
            ui->comboBox_b_frame->setEnabled(false);
            encodeSettings.b_frames=0;
        }
    }

    //GOPサイズ
    {
        QStringList gop_items;
        gop_items <<"1"<<"15"<<"30"<<"60"<<"120"<<"250"<<"300";
        ui->comboBox_gop->addItems(gop_items);
        QObject::connect(ui->comboBox_gop, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.gop_size = settingmap[index].gop_items;
            combo_index_control2();
        }, Qt::QueuedConnection);
    }

    //詳細設定
    //プリセット
    {
        QStringList preset_items;
        preset_items<< "p1 (最速・低品質)"<< "p2"<< "p3"<< "p4"<< "p5"<< "p6"<< "p7 (最高品質・低速)";
        ui->comboBox_preset->addItems(preset_items);
        QObject::connect(ui->comboBox_preset, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.preset = settingmap[index].preset_items;
        }, Qt::QueuedConnection);
    }

    //最適化プロファイル（チューニング）
    {
        QStringList profile_items;
        profile_items<< "default (汎用)"<< "hq (高品質)"<< "ll (低遅延)"<< "ull (超低遅延)"<< "lossless (可逆圧縮)";
        ui->comboBox_profile->addItems(profile_items);
        QObject::connect(ui->comboBox_profile, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.tune = settingmap[index].profile_items;
        }, Qt::QueuedConnection);
    }

    // ターゲットビットレートスライダー
    ui->horizontalSlider_targetbitrate->setRange(1, 1000);
    QObject::connect(ui->horizontalSlider_targetbitrate, &QSlider::valueChanged, this, [&](int value){
        target_bit_rate = value;
        ui->label_targetbitrare_value->setText(QString::number(value) + " Mbps");
    }, Qt::QueuedConnection);

    // 最大ビットレートスライダー
    ui->horizontalSlider_maxbitrate->setRange(1, 1000);
    QObject::connect(ui->horizontalSlider_maxbitrate, &QSlider::valueChanged, this, [&](int value){
        max_bit_rate = value;
        ui->label_maxbitrate_value->setText(QString::number(value) + " Mbps");

        // ターゲットスライダーの範囲を更新
        int target = target_bit_rate;

        if(encodeSettings.rc_mode=="vbr"){
            ui->horizontalSlider_targetbitrate->setMaximum(value);
        }

        // 現在のターゲットが最大を超えた場合は調整
        if(target > value&&encodeSettings.rc_mode=="vbr"){
            target = value;
            target_bit_rate = target;
            ui->horizontalSlider_targetbitrate->setValue(target);
        }
    }, Qt::QueuedConnection);

    //cq
    ui->horizontalSlider_cq->setRange(1, 51);
    QObject::connect(ui->horizontalSlider_cq, &QSlider::valueChanged, this, [&](int value){
        encodeSettings.cq = value;
        ui->label_cq_value->setText(QString::number(value));
    }, Qt::QueuedConnection);


    //可変ビットレート
    {
        QStringList rc_items;
        rc_items << "CBR" << "VBR" << "CQ" ;
        ui->comboBox_rc->addItems(rc_items);
        QObject::connect(ui->comboBox_rc, &QComboBox::currentIndexChanged,this, [&](int index) {
            switch (index) {
                case 0:
                    encodeSettings.rc_mode = settingmap[index].rc_items;
                    ui->horizontalSlider_targetbitrate->setEnabled(true);
                    ui->label_targetbitrate->setEnabled(true);
                    ui->label_targetbitrare_value->setEnabled(true);
                    ui->horizontalSlider_maxbitrate->setEnabled(false);
                    ui->label_maxbitrate->setEnabled(false);
                    ui->label_maxbitrate_value->setEnabled(false);
                    ui->horizontalSlider_cq->setEnabled(false);
                    ui->label_cq_value->setEnabled(false);
                    ui->label_cq->setEnabled(false);
                    ui->horizontalSlider_targetbitrate->setRange(1,1000);
                    break;
                case 1:
                    encodeSettings.rc_mode = settingmap[index].rc_items;
                    ui->horizontalSlider_targetbitrate->setEnabled(true);
                    ui->label_targetbitrate->setEnabled(true);
                    ui->label_targetbitrare_value->setEnabled(true);
                    ui->horizontalSlider_maxbitrate->setEnabled(true);
                    ui->label_maxbitrate->setEnabled(true);
                    ui->label_maxbitrate_value->setEnabled(true);
                    ui->horizontalSlider_cq->setEnabled(false);
                    ui->label_cq_value->setEnabled(false);
                    ui->label_cq->setEnabled(false);
                    ui->horizontalSlider_targetbitrate->setRange(1,max_bit_rate);
                    break;
                case 2:
                    encodeSettings.rc_mode = settingmap[index].rc_items;
                    ui->horizontalSlider_targetbitrate->setEnabled(false);
                    ui->label_targetbitrate->setEnabled(false);
                    ui->label_targetbitrare_value->setEnabled(false);
                    ui->horizontalSlider_maxbitrate->setEnabled(false);
                    ui->label_maxbitrate->setEnabled(false);
                    ui->label_maxbitrate_value->setEnabled(false);
                    ui->horizontalSlider_cq->setEnabled(true);
                    ui->label_cq_value->setEnabled(true);
                    ui->label_cq->setEnabled(true);
                    break;
            }
        }, Qt::QueuedConnection);
    }

    //エンコードパス
    {
        QStringList pass_items;
        pass_items << "1 pass" << "2 pass quarter" << "2 pass full" ;
        ui->comboBox_encodepass->addItems(pass_items);
        QObject::connect(ui->comboBox_encodepass, &QComboBox::currentIndexChanged,this, [&](int index) {
            encodeSettings.pass_mode = settingmap[index].pass_items;
        }, Qt::QueuedConnection);
    }

    //エンコードスタート
    QObject::connect(ui->encodeStart_pushbutton, &QPushButton::clicked, this, [&]() {
        //CBRの場合は最大ビットレートもターゲットビットレートを入れる
        if(encodeSettings.rc_mode=="cbr"){
            encodeSettings.max_bit_rate=target_bit_rate*1000*1000;
            encodeSettings.target_bit_rate=target_bit_rate*1000*1000;
        }else{
            encodeSettings.max_bit_rate=max_bit_rate*1000*1000;
            encodeSettings.target_bit_rate=target_bit_rate*1000*1000;
        }

        //コーデックの最大のビットレートを超えた場合の制限
        if(encodeSettings.codec=="h264_nvenc"){
            if(encodeSettings.max_bit_rate>960000000){
                encodeSettings.max_bit_rate = 960000000;
            }
            if(encodeSettings.target_bit_rate>960000000){
                encodeSettings.target_bit_rate = 960000000;
            }
        }
        if(encodeSettings.codec=="hevc_nvenc"&&encodeSettings.max_bit_rate>800000000){
            if(encodeSettings.max_bit_rate>800000000){
                encodeSettings.max_bit_rate = 800000000;
            }
            if(encodeSettings.target_bit_rate>800000000){
                encodeSettings.target_bit_rate = 800000000;
            }
        }

        //パスが存在するか確認
        QFileInfo info(encodeSettings.encode_path);
        QString dirPath = info.absolutePath();
        QDir dir(dirPath);

        if (!dir.exists()) {
            QMessageBox::warning(this,
                                 tr("エンコードエラー"),
                                 tr("エンコード保存パスがありません"),
                                 QMessageBox::Ok);
            return;
        }

        //上書きできるか
        if(VideoInfo.Path==encodeSettings.encode_path){
            encodeSettings.encode_path = file_check(encodeSettings.encode_path);
            ui->label_filepath->setText(encodeSettings.encode_path);
        }

        //GPU、タイル設定を保存
        encodeSettings.tile_gpu_map = buildTileGpuMap();

        //エンコード開始
        emit signal_encode_start();

        //UI制御
        ui->encodeStart_pushbutton->setEnabled(false);
        ui->encodecancel_pushButton->setEnabled(true);
        ui->filepath_groupBox->setEnabled(false);
        ui->normalsetting_groupBox->setEnabled(false);
        ui->advancesetting_groupBox->setEnabled(false);
        multiGPUtabel_UI_Control(false);
        encode_flag=true;
    }, Qt::QueuedConnection);

    //エンコードキャンセル
    QObject::connect(ui->encodecancel_pushButton, &QPushButton::clicked, this, [&]() {
        emit signal_encode_stop();

        //UI制御
        ui->encodeStart_pushbutton->setEnabled(true);
        ui->encodecancel_pushButton->setEnabled(false);
        ui->filepath_groupBox->setEnabled(true);
        ui->normalsetting_groupBox->setEnabled(true);
        ui->advancesetting_groupBox->setEnabled(true);
        encode_flag=false;
    }, Qt::QueuedConnection);

    //iniファイルを元に初期値をUIに反映
    {
        iniFile_UI_Control();
        auto emitIndexChanged = [&](QComboBox* box){
            int idx = box->currentIndex();
            Q_EMIT box->currentIndexChanged(idx);
        };

        // 初期値セット
        ui->label_filepath->setText(encodeSettings.encode_path);

        ui->comboBox_codec->setCurrentIndex(combo_index[0]);
        emitIndexChanged(ui->comboBox_codec);

        ui->comboBox_splitencode->setCurrentIndex(combo_index[1]);
        emitIndexChanged(ui->comboBox_splitencode);

        ui->comboBox_b_frame->setCurrentIndex(combo_index[2]);
        emitIndexChanged(ui->comboBox_b_frame);

        ui->comboBox_preset->setCurrentIndex(combo_index[3]);
        emitIndexChanged(ui->comboBox_preset);

        ui->comboBox_profile->setCurrentIndex(combo_index[4]);
        emitIndexChanged(ui->comboBox_profile);

        ui->comboBox_rc->setCurrentIndex(combo_index[5]);
        emitIndexChanged(ui->comboBox_rc);

        ui->comboBox_encodepass->setCurrentIndex(combo_index[6]);
        emitIndexChanged(ui->comboBox_encodepass);

        ui->comboBox_gop->setCurrentIndex(combo_index[7]);
        emitIndexChanged(ui->comboBox_gop);

        ui->comboBox_tile->setCurrentIndex(combo_index[8]);
        emitIndexChanged(ui->comboBox_tile);

        ui->horizontalSlider_targetbitrate->setValue(target_bit_rate);
        ui->label_targetbitrare_value->setText(QString::number(target_bit_rate) + " Mbps");

        ui->horizontalSlider_maxbitrate->setValue(max_bit_rate);
        ui->label_maxbitrate_value->setText(QString::number(max_bit_rate) + " Mbps");

        ui->horizontalSlider_cq->setValue(encodeSettings.cq);
        ui->label_cq_value->setText(QString::number(encodeSettings.cq));
    }
}

encode_setting::~encode_setting()
{
    delete ui;
}

//初期値をUIに反映する(Combobox)
void encode_setting::iniFile_UI_Control(){
    // 読み込み
    // デフォルト
    combo_index[0] = foundIndex("codec", "h264_nvenc");
    bool foundPrimary = false;
    bool primaryAv1 = false;
    for (const auto &gpu : g_GPUInfo)
    {
        if (gpu.openglEnable)
        {
            foundPrimary = true;
            primaryAv1 = (gpu.CC_major > 8 || (gpu.CC_major == 8 && gpu.CC_minor >= 9));
            break;
        }
    }
    if (primaryAv1)
    {
        // primaryがAV1対応ならそのまま反映
        combo_index[0] = foundIndex("codec", encodeSettings.codec);
    }
    else
    {
        // primaryがAV1非対応、または見つからない場合
        if (encodeSettings.codec == "av1_nvenc")
            combo_index[0] = foundIndex("codec", "hevc_nvenc");
        else
            combo_index[0] = foundIndex("codec", encodeSettings.codec);
    }
    combo_index[1]=foundIndex("split_encode_mode",encodeSettings.split_encode_mode);
    combo_index[2]=foundIndex("b_frames",QString::number(encodeSettings.b_frames));
    combo_index[3]=foundIndex("preset",encodeSettings.preset);
    combo_index[4]=foundIndex("tune",encodeSettings.tune);
    combo_index[5]=foundIndex("rc_mode",encodeSettings.rc_mode);
    combo_index[6]=foundIndex("pass_mode",encodeSettings.pass_mode);
    combo_index[7]=foundIndex("gop_size",QString::number(encodeSettings.gop_size));
    combo_index[8]=foundIndex("encode_tile",QString::number(encodeSettings.encode_tile));
    target_bit_rate = encodeSettings.target_bit_rate/1000000;
    max_bit_rate = encodeSettings.max_bit_rate/1000000;
}

//comboボックスのインデックスを取得
int encode_setting::foundIndex(QString key,const QString& item){
    for (auto it = settingmap.constBegin(); it != settingmap.constEnd(); ++it){
        if (key=="codec"&&it.value().codec == item){
            return it.key();
        }else if(key=="split_encode_mode"&&it.value().splitencode_items == item){
            return it.key();
        }else if(key=="b_frames"&&it.value().B_frame_items == item.toInt()){
            return it.key();
        }else if(key=="preset"&&it.value().preset_items == item){
            return it.key();
        }else if(key=="tune"&&it.value().profile_items == item){
            return it.key();
        }else if(key=="rc_mode"&&it.value().rc_items == item){
            return it.key();
        }else if(key=="pass_mode"&&it.value().pass_items == item){
            return it.key();
        }else if(key=="gop_size"&&it.value().gop_items == item.toInt()){
            return it.key();
        }else if(key=="encode_tile"&&it.value().encode_tile == item.toInt()){
            return it.key();
        }
    }
    return -1;
}

//ファイルの重複チェック
QString encode_setting::file_check(const QString &filePath)
{
    QFileInfo fileInfo(filePath);
    QString dir = fileInfo.absolutePath(); // 例: "D:/"
    QString baseName = fileInfo.completeBaseName(); // 例: "test1(2)"
    QString ext = fileInfo.suffix();

    QString baseNameOnly = baseName;
    int counter = 1;

    QRegularExpression re(R"(^(.*)\((\d+)\)$)");
    QRegularExpressionMatch match = re.match(baseName);

    if (match.hasMatch()) {
        baseNameOnly = match.captured(1);
        counter = match.captured(2).toInt();
    }

    QString newPath = filePath;
    QDir qDir(dir);

    // ファイルが存在する間ループ
    while (QFile::exists(newPath)) {
        QString newFileName = QString("%1(%2).%3")
                                  .arg(baseNameOnly)
                                  .arg(counter++)
                                  .arg(ext);

        newPath = qDir.filePath(newFileName);
    }
    return newPath;
}

void encode_setting::closeEvent(QCloseEvent *event)
{
    if (encode_flag) {
        // ★ 処理中の場合は、閉じる操作を「無視」する
        qDebug() << "処理中のため、ウィンドウを閉じるのを拒否しました。";
        event->ignore();
    } else {
        // 処理中でなければ、通常通り閉じる
        event->accept();
        emit signal_encode_finished();
    }
}

//エンコード終了(最後までやって終了)
void encode_setting::encode_end(QString encode_time){
    // Qtのメインスレッドで警告ポップアップを表示
    QMetaObject::invokeMethod(this, [this,encode_time]() {
        QMessageBox::information(this,
                             tr("エンコード終了"),
                             tr("処理時間:\n%1").arg(encode_time),
                             QMessageBox::Ok);
    }, Qt::QueuedConnection);

    //UI制御
    progress_bar(0);
    ui->encodeStart_pushbutton->setEnabled(true);
    ui->encodecancel_pushButton->setEnabled(false);
    ui->filepath_groupBox->setEnabled(true);
    ui->normalsetting_groupBox->setEnabled(true);
    ui->advancesetting_groupBox->setEnabled(true);
    if(g_GPUInfo.size() > 1){
        multiGPUtabel_UI_Control(true);
        setGPUTable();
    }
    encode_flag=false;
}

//エンコード開始時、最大フレーム数などを取得
void encode_setting::slider(int min,int max){
    ui->encode_progressBar->setRange(min, max);
    ui->encode_progressBar->setValue(min);

    fps_cmbobox_control();
    tile_index_control();
    setGPUTable();
}

//進捗バーを動かす
void encode_setting::progress_bar(int value){
    ui->encode_progressBar->setValue(value);
}

//fps設定
void encode_setting::fps_cmbobox_control(){
    ui->comboBox_framerate->clear();

    //保存フレームレート
    {
        QStringList framerate_items;
        framerate_items << "1" << "2" << "5" << "10" << "15" << "20"
                        << "24" << "30" << "60" << "120" << "240" << "300";

        // fps を追加（未登録なら）
        QString fpsStr = QString::number(VideoInfo.fps);
        if (!framerate_items.contains(fpsStr))
            framerate_items << fpsStr;

        // 数値ソート
        std::sort(framerate_items.begin(), framerate_items.end(),
                  [](const QString& a, const QString& b) {
                      return a.toDouble() < b.toDouble();
                  });

        // ComboBox 初期化
        ui->comboBox_framerate->blockSignals(true); // signal防止
        ui->comboBox_framerate->clear();
        ui->comboBox_framerate->addItems(framerate_items);

        QObject::connect(ui->comboBox_framerate,&QComboBox::currentIndexChanged,this,[&](int index) {
                             double fps = ui->comboBox_framerate->currentText().toDouble();
                             encodeSettings.save_fps = fps;
        });

        // 初期値を VideoInfo.fps に設定
        ui->comboBox_framerate->setCurrentText(fpsStr);
        ui->comboBox_framerate->blockSignals(false);
        encodeSettings.save_fps=VideoInfo.fps;
    }
}

//コーデックに応じて非対応オプションを非表示
void encode_setting::combo_index_control2(){
    if(encodeSettings.gop_size == 1){
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),1,7,true,true);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,4,false,false);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,1,true,true);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),4,4,true,true);
    }else if(encodeSettings.encode_tile>1){
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),1,7,true,true);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,4,false,false);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),4,4,true,true);
        //combo_index_control(ui->comboBox_splitencode,qobject_cast<QListView*>(ui->comboBox_splitencode->view()),1,1,true,true);
    }else if(encodeSettings.codec == "h264_nvenc"){
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),5,7,true,true);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,4,false,false);
        //combo_index_control(ui->comboBox_splitencode,qobject_cast<QListView*>(ui->comboBox_splitencode->view()),1,1,false,false);
    }else if(encodeSettings.codec == "hevc_nvenc"){
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),6,7,true,true);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,4,false,false);
        //combo_index_control(ui->comboBox_splitencode,qobject_cast<QListView*>(ui->comboBox_splitencode->view()),1,1,false,false);
    }else if(encodeSettings.codec == "av1_nvenc"){
        combo_index_control(ui->comboBox_b_frame,qobject_cast<QListView*>(ui->comboBox_b_frame->view()),0,7,false,false);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),0,4,false,false);
        combo_index_control(ui->comboBox_profile,qobject_cast<QListView*>(ui->comboBox_profile->view()),4,4,true,true);
        //combo_index_control(ui->comboBox_splitencode,qobject_cast<QListView*>(ui->comboBox_splitencode->view()),1,1,false,false);
    }
}

//特定のコンボボックスを非表示にする
void encode_setting::combo_index_control(QComboBox* comboBox,QListView* view, int ini_index, int target_index, bool flag,bool index_change_flag) {
    for (int i = ini_index; i <= target_index; i++) {
        view->setRowHidden(i, flag);
    }

    if (flag) {
        int currentIndex = comboBox->currentIndex();

        if ((currentIndex >= ini_index&&index_change_flag)&&(currentIndex <= target_index&&index_change_flag)) {
            int change_index=ini_index-1;
            if(change_index>=0){
                comboBox->setCurrentIndex(ini_index-1);
            }else{
                comboBox->setCurrentIndex(target_index+1);
            }

            // qDebug()<<currentIndex<<":"<<ini_index<<":"<<target_index<<":"<<change_index;
        }
    }
}

//タイル数コントロール
void encode_setting::tile_index_control(){
    //H264チェック
    if(VideoInfo.width*VideoInfo.width_scale/encodeSettings.width_tile>4096||VideoInfo.height*VideoInfo.height_scale/encodeSettings.height_tile>4096){
        qobject_cast<QListView*>(ui->comboBox_codec->view())->setRowHidden(0,true);
        ui->comboBox_codec->setCurrentIndex(1);
    }else{
        qobject_cast<QListView*>(ui->comboBox_codec->view())->setRowHidden(0,false);
    }

    //タイル数チェック
    qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(0,false);
    qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(1,false);
    qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(2,false);
    qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(3,false);
    //2×2
    if(VideoInfo.width*VideoInfo.width_scale/2>8192||VideoInfo.height*VideoInfo.height_scale/2>8192){
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(0,true);
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(1,true);
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(2,true);
        if(ui->comboBox_tile->currentIndex()<3){
            ui->comboBox_tile->setCurrentIndex(3);
        }
    }

    //2×1
    if(VideoInfo.width*VideoInfo.width_scale/2>8192||VideoInfo.height*VideoInfo.height_scale>8192){
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(0,true);
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(1,true);
        if(ui->comboBox_tile->currentIndex()<2){
            ui->comboBox_tile->setCurrentIndex(2);
        }
    }

    //1×1
    if(VideoInfo.width*VideoInfo.width_scale>8192||VideoInfo.height*VideoInfo.height_scale>8192){
        qobject_cast<QListView*>(ui->comboBox_tile->view())->setRowHidden(0,true);
        if(ui->comboBox_tile->currentIndex()<1){
            ui->comboBox_tile->setCurrentIndex(1);
        }
    }
}

//タイル数から縦横分割数を変換
void encode_setting::tile_split_exchange(){
    int max_size = encodeSettings.encode_tile;
    if(max_size==2){
        encodeSettings.width_tile = 2;
        encodeSettings.height_tile  = 1;
    }else if(max_size==4){
        encodeSettings.width_tile = 2;
        encodeSettings.height_tile  = 2;
    }else if(max_size==8){
        encodeSettings.width_tile = 4;
        encodeSettings.height_tile  = 2;
    }else{
        encodeSettings.width_tile = 1;
        encodeSettings.height_tile  = 1;
    }
}

//マルチGPU設定表作成
void encode_setting::setGPUTable()
{
    ui->tableWidget_gpu->blockSignals(true);
    ui->tableWidget_gpu->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableWidget_gpu->setColumnCount(5);
    ui->tableWidget_gpu->setRowCount((int)g_GPUInfo.size());

    restoreGpuRestriction();     // AV1以外なら復元
    applyAv1GpuRestriction();    // AV1なら制限適用

    QStringList headers;
    headers << "GPU Name" << "Memory(MB)" << "Detail" << "Weight" << "Tiles";
    ui->tableWidget_gpu->setHorizontalHeaderLabels(headers);

    std::vector<int> order;

    for (int i = 0; i < (int)g_GPUInfo.size(); i++)
        if (g_GPUInfo[i].openglEnable)
            order.push_back(i);

    for (int i = 0; i < (int)g_GPUInfo.size(); i++)
        if (!g_GPUInfo[i].openglEnable)
            order.push_back(i);

    for (int row = 0; row < (int)order.size(); row++)
    {
        const auto &gpu = g_GPUInfo[order[row]];

        // GPU Name
        auto *nameItem = new QTableWidgetItem(
            gpu.deviceName + "(" + QString::number(gpu.deviceID) + ")");
        nameItem->setData(Qt::UserRole, gpu.deviceID); // ★deviceID保持
        ui->tableWidget_gpu->setItem(row, 0, nameItem);

        // Memory
        ui->tableWidget_gpu->setItem(row, 1,
                                     new QTableWidgetItem(QString("%1 / %2")
                                                              .arg(gpu.Memory_Usage)
                                                              .arg(gpu.Max_Memory_Usage)));

        // Detail
        ui->tableWidget_gpu->setItem(row, 2,
                                     new QTableWidgetItem(gpu.openglEnable ? "Primary GPU" : ""));

        // Weight SpinBox
        QSpinBox *spin = new QSpinBox();

        // Primary GPUは必ず使う
        if (gpu.openglEnable)
            spin->setRange(1, 10);
        else
            spin->setRange(0, 10);

        spin->setValue(gpu.tile_weight);
        ui->tableWidget_gpu->setCellWidget(row, 3, spin);

        // Tiles 表示
        ui->tableWidget_gpu->setItem(row, 4, new QTableWidgetItem(""));
        ui->tableWidget_gpu->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);

        connect(spin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &encode_setting::updateGpuTileAllocation);
    }

    ui->tableWidget_gpu->blockSignals(false);

    updateGpuTileAllocation();
    ui->tableWidget_gpu->resizeColumnsToContents();
    ui->tableWidget_gpu->setColumnWidth(3, 50);

    // GPUが1枚しかない場合はグレーアウト
    if(g_GPUInfo.size() <= 1){
        multiGPUtabel_UI_Control(false);
    }
}

//AV1対応可否
bool encode_setting::isAv1Supported(const GPUInfo& gpu)
{
    return (gpu.CC_major > 8) || (gpu.CC_major == 8 && gpu.CC_minor >= 9);
}

//セカンダリGPU以降AV1対応可否
void encode_setting::applyAv1GpuRestriction()
{
    if (encodeSettings.codec != "av1_nvenc")
        return;

    for (auto &gpu : g_GPUInfo)
    {
        if (gpu.openglEnable)
        {
            if (!isAv1Supported(gpu))
                return; // primaryが非対応なら何もしない（そもそもAV1出すべきではない）
            break;
        }
    }

    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        auto *nameItem = ui->tableWidget_gpu->item(row, 0);
        if (!nameItem) continue;

        int deviceID = nameItem->data(Qt::UserRole).toInt();

        GPUInfo *gpuPtr = nullptr;
        for (auto &g : g_GPUInfo)
        {
            if (g.deviceID == deviceID) { gpuPtr = &g; break; }
        }
        if (!gpuPtr) continue;

        // primaryは除外
        if (gpuPtr->openglEnable)
            continue;

        // secondaryでAV1非対応なら無効化
        if (!isAv1Supported(*gpuPtr))
        {
            QSpinBox *spin =
                qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));

            if (spin)
            {
                // 初回だけ退避
                if (gpuPtr->backup_tile_weight < 0)
                    gpuPtr->backup_tile_weight = spin->value();

                gpuPtr->av1_forced_disabled = true;
                spin->blockSignals(true);
                spin->setValue(0);
                spin->setEnabled(false);
                spin->setStyleSheet("QSpinBox { color: gray; }");
                spin->blockSignals(false);
            }

            gpuPtr->tile_weight = 0;

            ui->tableWidget_gpu->item(row, 4)->setText("Not Used (AV1 Unsupported)");

            for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
            {
                QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
                if (it) it->setForeground(Qt::gray);
            }
        }
    }
}

//AV1で無効化したUIを戻す
void encode_setting::restoreGpuRestriction()
{
    if (encodeSettings.codec == "av1_nvenc")
        return;

    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        auto *nameItem = ui->tableWidget_gpu->item(row, 0);
        if (!nameItem) continue;

        int deviceID = nameItem->data(Qt::UserRole).toInt();

        GPUInfo *gpuPtr = nullptr;
        for (auto &g : g_GPUInfo)
        {
            if (g.deviceID == deviceID) { gpuPtr = &g; break; }
        }
        if (!gpuPtr) continue;

        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));
        if (!spin) continue;

        // backupがあるなら復元
        if (gpuPtr->backup_tile_weight >= 0)
        {
            spin->blockSignals(true);
            spin->setEnabled(true);
            spin->setStyleSheet("");
            spin->setValue(gpuPtr->backup_tile_weight);
            spin->blockSignals(false);

            gpuPtr->tile_weight = gpuPtr->backup_tile_weight;
            gpuPtr->backup_tile_weight = -1;
            gpuPtr->av1_forced_disabled = false;
        }
    }
}

//マルチGPU設定タイル割り当て
void encode_setting::updateGpuTileAllocation()
{
    ui->tableWidget_gpu->blockSignals(true);

    int totalTiles = encodeSettings.encode_tile;
    if (totalTiles <= 0) totalTiles = 1;

    int totalWeight = 0;

    // Weight合計（weight>0のみ対象）
    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));
        if (!spin) continue;

        if (spin->value() <= 0)
            continue;

        totalWeight += spin->value();
    }

    if (totalWeight <= 0) totalWeight = 1;

    int usedTiles = 0;

    // タイル計算
    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));
        if (!spin) continue;

        int weight = spin->value();

        // weight=0 → 無効
        if (weight <= 0)
        {
            ui->tableWidget_gpu->item(row, 4)->setText("Not Used");

            // グレーアウト
            for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
            {
                QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
                if (it) it->setForeground(Qt::gray);
            }

            continue;
        }

        int tiles = (weight * totalTiles) / totalWeight;
        usedTiles += tiles;
        ui->tableWidget_gpu->item(row, 4)->setData(Qt::UserRole, tiles);
        ui->tableWidget_gpu->item(row, 4)->setText(QString::number(tiles));

        // 有効なので色を戻す
        for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
        {
            QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
            if (it) it->setForeground(Qt::black);
        }
    }

    // 余り配布（weight>0 の行だけに配る）
    int remain = totalTiles - usedTiles;

    for (int row = 0; row < ui->tableWidget_gpu->rowCount() && remain > 0; row++)
    {
        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));
        if (!spin) continue;

        if (spin->value() <= 0)
            continue;

        int current = ui->tableWidget_gpu->item(row, 4)->data(Qt::UserRole).toInt();
        current++;
        ui->tableWidget_gpu->item(row, 4)->setData(Qt::UserRole, current);

        remain--;
    }

    // 最終チェック：tiles=0ならNot Used表示にする
    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        //デバイスID取得
        auto *nameItem = ui->tableWidget_gpu->item(row, 0);
        if (!nameItem) continue;
        int deviceID = nameItem->data(Qt::UserRole).toInt();

        // g_GPUInfo の該当GPUを探す
        GPUInfo* gpuPtr = nullptr;
        for (auto &g : g_GPUInfo)
        {
            if (g.deviceID == deviceID)
            {
                gpuPtr = &g;
                break;
            }
        }
        if (!gpuPtr) continue;

        //spinbox値取得
        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));
        if (!spin) continue;
        int weight = spin->value();
        gpuPtr->tile_weight = weight;

        int tiles = ui->tableWidget_gpu->item(row, 4)->data(Qt::UserRole).toInt();
        int percent = int((double(tiles) / double(totalTiles)) * 100.0 + 0.5);

        // 最初に必ず初期化
        spin->setEnabled(true);
        spin->setStyleSheet("");

        // ★ AV1強制無効を最優先
        if (gpuPtr->av1_forced_disabled)
        {
            ui->tableWidget_gpu->item(row, 4)->setText("Not Used (AV1 Unsupported)");

            spin->blockSignals(true);
            spin->setValue(0);
            spin->setEnabled(false);
            spin->setStyleSheet("QSpinBox { color: gray; }");
            spin->blockSignals(false);

            gpuPtr->tile_weight = 0;

            for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
            {
                QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
                if (it) it->setForeground(Qt::gray);
            }
            continue;
        }

        // ★ 手動でweight=0の場合
        if (weight <= 0)
        {
            ui->tableWidget_gpu->item(row, 4)->setText("Not Used");

            // 手動ならspinは触れるようにする
            spin->setEnabled(true);
            spin->setStyleSheet("");

            for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
            {
                QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
                if (it) it->setForeground(Qt::gray);
            }

            gpuPtr->tile_weight = 0;
            continue;
        }

        // tiles=0 自動無効
        if (tiles <= 0 && encodeSettings.encode_tile < g_GPUInfo.size())
        {
            ui->tableWidget_gpu->item(row, 4)->setText("0 (0%)");

            spin->setEnabled(false);
            spin->setStyleSheet("QSpinBox { color: gray; }");

            for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
            {
                QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
                if (it) it->setForeground(Qt::gray);
            }
            continue;
        }

        // tiles>0 正常
        ui->tableWidget_gpu->item(row, 4)->setText(
            QString("%1 (%2%)").arg(tiles).arg(percent)
            );

        for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
        {
            QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
            if (it) it->setForeground(Qt::black);
        }
    }

    ui->tableWidget_gpu->blockSignals(false);
}

//割り当てたタイルを配列に格納
std::vector<int> encode_setting::buildTileGpuMap()
{
    std::vector<int> tile_gpu_map;

    int totalTiles = encodeSettings.encode_tile;
    if (totalTiles <= 0) totalTiles = 1;

    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        auto *nameItem = ui->tableWidget_gpu->item(row, 0);
        if (!nameItem) continue;

        int deviceID = nameItem->data(Qt::UserRole).toInt();

        auto *tileItem = ui->tableWidget_gpu->item(row, 4);
        if (!tileItem) continue;

        int tiles = tileItem->data(Qt::UserRole).toInt(); // ★tiles数

        for (int t = 0; t < tiles; t++)
            tile_gpu_map.push_back(deviceID);
    }

    // 念のため不足補正
    while ((int)tile_gpu_map.size() < totalTiles)
        tile_gpu_map.push_back(g_openglDeviceID);

    // 超えたら切る
    if ((int)tile_gpu_map.size() > totalTiles)
        tile_gpu_map.resize(totalTiles);

    return tile_gpu_map;
}

//マルチgpu UI制御
void encode_setting::multiGPUtabel_UI_Control(bool flag)
{
    bool disable = (!flag);

    for (int row = 0; row < ui->tableWidget_gpu->rowCount(); row++)
    {
        for (int col = 0; col < ui->tableWidget_gpu->columnCount(); col++)
        {
            QTableWidgetItem *it = ui->tableWidget_gpu->item(row, col);
            if (!it) continue;

            if (disable)
                it->setForeground(QBrush(Qt::gray));
            else
                it->setForeground(QBrush(Qt::black));
        }

        QSpinBox *spin =
            qobject_cast<QSpinBox*>(ui->tableWidget_gpu->cellWidget(row, 3));

        if (spin)
        {
            if (disable)
            {
                spin->setEnabled(false);
                spin->setStyleSheet("QSpinBox { color: gray; }");
            }
            else
            {
                spin->setEnabled(true);
                spin->setStyleSheet("");   // ★元に戻す
            }
        }
    }

    ui->tableWidget_gpu->setEnabled(!disable);
    ui->label_GPUTable->setEnabled(!disable);
}
