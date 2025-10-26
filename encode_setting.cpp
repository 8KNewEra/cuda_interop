#include "encode_setting.h"
#include "qevent.h"
#include "ui_encode_setting.h"

encode_setting::encode_setting(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::encode_setting)
{
    ui->setupUi(this);

    //設定ファイルが無ければ作成
    init_txt();

    settingmap[0] = {"h264_nvenc", 1,"0",0,"p1","default","cbr","1pass",1};
    settingmap[1] = {"hevc_nvenc",2,"1",1,"p2","hq","vbr","2pass-quarter-res",15};
    settingmap[2] = {"av1_nvenc",5,"",2,"p3","ll","","2pass-full-res",30};
    settingmap[3] = {"",10,"",3,"p4","ull","","",60};
    settingmap[4] = {"",15,"",4,"p5","lossless","","",120};
    settingmap[5] = {"",20,"",5,"p6","","","",250};
    settingmap[6] = {"",30,"",6,"p7","","","",30};
    settingmap[7] = {"",60,"",7,"p8","","","",0};
    settingmap[8] = {"",120,"",8,"","","","",0};
    settingmap[9] = {"",240,"",9,"","","","",0};
    settingmap[10] = {"",300,"",10,"","","","",0};

    //ファイルパス
    {
        QObject::connect(ui->filesave_pushButton, &QPushButton::clicked, this, [&]() {
            QString filter = "MP4 Files (*.mp4);;All Files (*.*)";
            QString fileName = QFileDialog::getSaveFileName(
                this,
                tr("名前を付けて保存"),
                "C:/",        // 初期ディレクトリ
                filter
                );

            if (!fileName.isEmpty()) {
                ui->label_filepath->setText(fileName);
                settings.Save_Path=fileName.toStdString();
            }
        }, Qt::QueuedConnection);
    }

    //コーデック
    {
        QStringList codec_items;
        codec_items << "H.264" << "H.265" << "AV1";
        ui->comboBox_codec->addItems(codec_items);
        QObject::connect(ui->comboBox_codec, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.Codec= settingmap[index].codec.toStdString();
        }, Qt::QueuedConnection);
    }

    //保存フレームレート
    {
        QStringList framerate_items;
        framerate_items << "1" << "2" << "5" << "10" << "15" << "20" << "30" << "60" << "120" << "240" << "300";
        ui->comboBox_framerate->addItems(framerate_items);
        QObject::connect(ui->comboBox_framerate, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.save_fps = settingmap[index].framerate_items;
        }, Qt::QueuedConnection);
    }

    //スプリットエンコード
    {
        QStringList splitencode_items;
        splitencode_items << "Disable" << "Enable" ;
        ui->comboBox_splitencode->addItems(splitencode_items);
        QObject::connect(ui->comboBox_splitencode, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.split_encode_mode=settingmap[index].splitencode_items.toStdString();
        }, Qt::QueuedConnection);
    }

    //Bフレーム
    {
        QStringList B_frame_items;
        B_frame_items << "0" << "1"<<"2"<<"3"<<"4"<<"5"<<"6"<<"7";
        ui->comboBox_b_frame->addItems(B_frame_items);
        QObject::connect(ui->comboBox_b_frame, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.b_frames = settingmap[index].B_frame_items;
        }, Qt::QueuedConnection);
    }

    //GOPサイズ
    {
        QStringList gop_items;
        gop_items <<"1"<<"15"<<"30"<<"60"<<"120"<<"250"<<"300";
        ui->comboBox_gop->addItems(gop_items);
        QObject::connect(ui->comboBox_gop, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.gop_size = settingmap[index].gop_items;
        }, Qt::QueuedConnection);
    }

    //詳細設定
    //プリセット
    {
        QStringList preset_items;
        preset_items<< "p1 (最速・低品質)"<< "p2"<< "p3"<< "p4"<< "p5"<< "p6"<< "p7 (最高品質・低速)";
        ui->comboBox_preset->addItems(preset_items);
        QObject::connect(ui->comboBox_preset, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.preset = settingmap[index].preset_items.toStdString();
        }, Qt::QueuedConnection);
    }

    //最適化プロファイル（チューニング）
    {
        QStringList profile_items;
        profile_items<< "default (汎用)"<< "hq (高品質)"<< "ll (低遅延)"<< "ull (超低遅延)"<< "lossless (可逆圧縮)";
        ui->comboBox_profile->addItems(profile_items);
        QObject::connect(ui->comboBox_profile, &QComboBox::currentIndexChanged, this, [&](int index) {
            settings.tune = settingmap[index].profile_items.toStdString();
        }, Qt::QueuedConnection);
    }

    // ターゲットビットレートスライダー
    ui->horizontalSlider_targetbitrate->setRange(1, max_bit_rate);
    QObject::connect(ui->horizontalSlider_targetbitrate, &QSlider::valueChanged, this, [&](int value){
        target_bit_rate = value;
        ui->label_targetbitrare_value->setText(QString::number(value) + " Mbps");
    }, Qt::QueuedConnection);

    // 最大ビットレートスライダー
    ui->horizontalSlider_maxbitrate->setRange(1, 200);
    QObject::connect(ui->horizontalSlider_maxbitrate, &QSlider::valueChanged, this, [&](int value){
        max_bit_rate = value;
        ui->label_maxbitrate_value->setText(QString::number(value) + " Mbps");

        // ターゲットスライダーの範囲を更新
        int target = target_bit_rate;
        ui->horizontalSlider_targetbitrate->setMaximum(value);

        // 現在のターゲットが最大を超えた場合は調整
        if(target > value){
            target = value;
            target_bit_rate = target;
            ui->horizontalSlider_targetbitrate->setValue(target);
        }
    }, Qt::QueuedConnection);


    //可変ビットレート
    {
        QStringList rc_items;
        rc_items << "CBR" << "VBR" ;
        ui->comboBox_rc->addItems(rc_items);
        QObject::connect(ui->comboBox_rc, &QComboBox::currentIndexChanged,this, [&](int index) {
            switch (index) {
                case 0:
                    settings.rc_mode = settingmap[index].rc_items.toStdString();
                    ui->horizontalSlider_maxbitrate->setEnabled(false);
                    ui->horizontalSlider_targetbitrate->setRange(1,200);
                    break;
                case 1:
                    settings.rc_mode = settingmap[index].rc_items.toStdString();
                    ui->horizontalSlider_maxbitrate->setEnabled(true);
                    ui->horizontalSlider_targetbitrate->setRange(1,max_bit_rate);
            }
        }, Qt::QueuedConnection);
    }

    //エンコードパス
    {
        QStringList pass_items;
        pass_items << "1 pass" << "2 pass" << "2 pass 高精度" ;
        ui->comboBox_encodepass->addItems(pass_items);
        QObject::connect(ui->comboBox_encodepass, &QComboBox::currentIndexChanged,this, [&](int index) {
            settings.pass_mode = settingmap[index].pass_items.toStdString();
        }, Qt::QueuedConnection);
    }

    //エンコードスタート
    QObject::connect(ui->encodeStart_pushbutton, &QPushButton::clicked, this, [&]() {
        //CBRの場合は最大ビットレートもターゲットビットレートを入れる
        if(settings.rc_mode=="cbr"){
            settings.max_bit_rate=target_bit_rate*1000*1000;
            settings.target_bit_rate=target_bit_rate*1000*1000;
        }else{
            settings.max_bit_rate=max_bit_rate*1000*1000;
            settings.target_bit_rate=target_bit_rate*1000*1000;
        }

        //上書きできるか
        if (!ui->allow_overwrite_checkBox->isChecked()||(VideoInfo.Path==settings.Save_Path)) {
            allow_overwrite = false;
            settings.Save_Path = makeUniqueFileName(QString::fromStdString(settings.Save_Path)).toStdString();
            ui->label_filepath->setText(QString::fromStdString(settings.Save_Path));
        }else{
            allow_overwrite = true;
        }

        //設定データ保存
        write_txt();

        //エンコード開始
        emit signal_encode_start();
        ui->encodeStart_pushbutton->setEnabled(false);
        ui->encodecancel_pushButton->setEnabled(true);
        encode_flag=true;
    }, Qt::QueuedConnection);

    //エンコードキャンセル
    QObject::connect(ui->encodecancel_pushButton, &QPushButton::clicked, this, [&]() {
        emit signal_encode_stop();
        ui->encodeStart_pushbutton->setEnabled(true);
        ui->encodecancel_pushButton->setEnabled(false);
        encode_flag=false;
    }, Qt::QueuedConnection);

    read_txt();

    {
        auto emitIndexChanged = [&](QComboBox* box){
            int idx = box->currentIndex();
            Q_EMIT box->currentIndexChanged(idx);
        };

        // 初期値セット
        ui->label_filepath->setText(QString::fromStdString(settings.Save_Path));

        ui->comboBox_codec->setCurrentIndex(combo_index[0]);
        emitIndexChanged(ui->comboBox_codec);

        ui->comboBox_framerate->setCurrentIndex(combo_index[1]);
        emitIndexChanged(ui->comboBox_framerate);

        ui->comboBox_splitencode->setCurrentIndex(combo_index[2]);
        emitIndexChanged(ui->comboBox_splitencode);

        ui->comboBox_b_frame->setCurrentIndex(combo_index[3]);
        emitIndexChanged(ui->comboBox_b_frame);

        ui->comboBox_preset->setCurrentIndex(combo_index[4]);
        emitIndexChanged(ui->comboBox_preset);

        ui->comboBox_profile->setCurrentIndex(combo_index[5]);
        emitIndexChanged(ui->comboBox_profile);

        ui->comboBox_rc->setCurrentIndex(combo_index[6]);
        emitIndexChanged(ui->comboBox_rc);

        ui->comboBox_encodepass->setCurrentIndex(combo_index[7]);
        emitIndexChanged(ui->comboBox_encodepass);

        ui->comboBox_gop->setCurrentIndex(combo_index[8]);
        emitIndexChanged(ui->comboBox_gop);

        ui->horizontalSlider_targetbitrate->setValue(target_bit_rate);
        ui->label_targetbitrare_value->setText(QString::number(target_bit_rate) + " Mbps");

        ui->horizontalSlider_maxbitrate->setValue(max_bit_rate);
        ui->label_maxbitrate_value->setText(QString::number(max_bit_rate) + " Mbps");

        ui->allow_overwrite_checkBox->setChecked(allow_overwrite);
    }
}

encode_setting::~encode_setting()
{
    delete ui;
}

void encode_setting::init_txt(){
    QFile file("setting.txt");
    if(!file.exists()){
        // 新規作成
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << "Save_Path:\"D:/test1.mp4\"\n";
            out << "Codec:\"h264_nvenc\"\n";
            out << "save_fps:30\n";
            out << "preset:\"p4\"\n";
            out << "tune:\"default\"\n";
            out << "b_frames:\"0\"\n";
            out << "gop_size:\"1\"\n";
            out << "split_encode_mode:\"0\"\n";
            out << "pass_mode:\"1pass\"\n";
            out << "rc_mode:\"cbr\"\n";
            out << "allow_overwrite:\"false\"\n";
            out << "target_bit_rate:100000000\n";
            out << "max_bit_rate:200000000\n";
            file.close();
            qDebug() << "新しい設定ファイルを作成しました。";
        }
    }
}

void encode_setting::read_txt(){
    QString filePath = "setting.txt";
    QFile file(filePath);

    // 読み込み
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            if (line.startsWith("Save_Path:")){
                settings.Save_Path = line.mid(line.indexOf(':') + 1).remove('"').toStdString();
            }else if (line.startsWith("Codec:")){
                settings.Codec = line.split(':')[1].remove('"').toStdString();
                combo_index[0]=foundIndex("codec",line.split(':')[1].remove('"'));
            }else if (line.startsWith("save_fps:")){
                settings.save_fps = line.split(':')[1].toInt();
                combo_index[1]=foundIndex("fps",line.split(':')[1].remove('"'));
            }else if (line.startsWith("split_encode_mode:")){
                settings.split_encode_mode = line.split(':')[1].toInt();
                combo_index[2]=foundIndex("split_encode_mode",line.split(':')[1].remove('"'));
            }else if (line.startsWith("b_frames:")){
                settings.b_frames = line.split(':')[1].toInt();
                combo_index[3]=foundIndex("b_frames",line.split(':')[1].remove('"'));
            }else if (line.startsWith("preset:")){
                settings.preset = line.split(':')[1].toInt();
                combo_index[4]=foundIndex("preset",line.split(':')[1].remove('"'));
            }else if (line.startsWith("tune:")){
                settings.tune = line.split(':')[1].toInt();
                combo_index[5]=foundIndex("tune",line.split(':')[1].remove('"'));
            }else if (line.startsWith("rc_mode:")){
                settings.rc_mode = line.split(':')[1].toInt();
                combo_index[6]=foundIndex("rc_mode",line.split(':')[1].remove('"'));
            }else if (line.startsWith("pass_mode:")){
                settings.pass_mode = line.split(':')[1].toInt();
                combo_index[7]=foundIndex("pass_mode",line.split(':')[1].remove('"'));
            }else if (line.startsWith("gop_size:")){
                settings.gop_size = line.split(':')[1].toInt();
                combo_index[8]=foundIndex("gop_size",line.split(':')[1].remove('"'));
            }else if (line.startsWith("allow_overwrite:")){
                allow_overwrite = line.split(':')[1].remove('"').trimmed().compare("true", Qt::CaseInsensitive) == 0;
            }else if (line.startsWith("target_bit_rate:")){
                settings.target_bit_rate = line.split(':')[1].remove('"').toInt();
                target_bit_rate = settings.target_bit_rate/1000000;
            }else if (line.startsWith("max_bit_rate:")){
                settings.max_bit_rate = line.split(':')[1].remove('"').toInt();
                max_bit_rate = settings.max_bit_rate/1000000;
            }
        }
        file.close();
    } else {
        qWarning() << "設定ファイルの読み込みに失敗しました。";
    }
}

int encode_setting::foundIndex(QString key,const QString& item){
    for (auto it = settingmap.constBegin(); it != settingmap.constEnd(); ++it){
        if (key=="codec"&&it.value().codec == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="fps"&&it.value().framerate_items == item.toInt()){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="split_encode_mode"&&it.value().splitencode_items == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="b_frames"&&it.value().B_frame_items == item.toInt()){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="preset"&&it.value().preset_items == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="tune"&&it.value().profile_items == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="rc_mode"&&it.value().rc_items == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="pass_mode"&&it.value().pass_items == item){
            qDebug()<<it.key();
            return it.key();
        }else if(key=="gop_size"&&it.value().gop_items == item.toInt()){
            qDebug()<<it.key();
            return it.key();
        }
    }

    return -1;
}

void encode_setting::write_txt(){
    QFile file("setting.txt");
    //既存データを書き込み
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << "Save_Path:\"" << QString::fromStdString(settings.Save_Path)<< "\"\n";
        out << "Codec:\"" << QString::fromStdString(settings.Codec) << "\"\n";
        out << "save_fps:\"" << QString::number(settings.save_fps) << "\"\n";
        out << "preset:\"" << QString::fromStdString(settings.preset) << "\"\n";
        out << "tune:\"" << QString::fromStdString(settings.tune) << "\"\n";
        out << "b_frames:\"" << QString::number(settings.b_frames) << "\"\n";
        out << "gop_size:\"" << QString::number(settings.gop_size) << "\"\n";
        out << "split_encode_mode:\"" << QString::fromStdString(settings.split_encode_mode) << "\"\n";
        out << "pass_mode:\"" << QString::fromStdString(settings.pass_mode) << "\"\n";
        out << "rc_mode:\"" << QString::fromStdString(settings.rc_mode) << "\"\n";
        out << "allow_overwrite:\"" << QVariant(allow_overwrite).toString() << "\"\n";
        out << "target_bit_rate:\"" << QString::number(settings.target_bit_rate) << "\"\n";
        out << "max_bit_rate:\"" << QString::number(settings.max_bit_rate) << "\"\n";
        file.close();
        qDebug() << "新しい設定ファイルを追加しました。";
    }
}

QString encode_setting::makeUniqueFileName(const QString &filePath)
{
    QFileInfo fileInfo(filePath);
    QString dir = fileInfo.absolutePath();
    QString baseName = fileInfo.completeBaseName(); // 拡張子を除いた部分
    QString ext = fileInfo.suffix();

    // baseName の末尾に (数字) が既にあるかチェック
    QRegularExpression re(R"(^(.*)\((\d+)\)$)");
    QRegularExpressionMatch match = re.match(baseName);
    if (match.hasMatch()) {
        baseName = match.captured(1); // "test1"
    }

    QString newPath = filePath;
    int counter = 1;

    // ファイルが存在する間ループ
    while (QFile::exists(newPath)) {
        newPath = QString("%1/%2(%3).%4")
        .arg(dir)
            .arg(baseName)
            .arg(counter++)
            .arg(ext);
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
void encode_setting::encode_end(){
    ui->encodeStart_pushbutton->setEnabled(true);
    ui->encodecancel_pushButton->setEnabled(false);
    encode_flag=false;
}

//最大フレーム数などを取得
void encode_setting::slider(int min,int max){
    ui->encode_progressBar->setRange(min, max);
    ui->encode_progressBar->setValue(min);
}

//進捗バーを動かす
void encode_setting::progress_bar(int value){
    ui->encode_progressBar->setValue(value);
}
