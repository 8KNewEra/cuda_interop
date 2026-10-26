#include "src/ui_control/encode_setting.h"
#include "qevent.h"
#include "qspinbox.h"
#include "ui_encode_setting.h"

namespace {
// 詳細設定の行配置。.ui が固定座標なので座標を自分で詰める。
//   基準の行を隠したときはスライダー行がその位置から始まる。
const int kBasisRowTop = 127;
const int kRateRowTop  = 160;
const int kRateRowStep = 31;
const int kRowHeight   = 22;    // スライダー 1 行の高さ
const int kGroupPad    = 12;    // グループ下端の余白

// 情報行（ビットレート換算と VRAM）
const int kInfoGap    = 8;
const int kInfoStep   = 18;
const int kInfoHeight = 16;

// 音声グループの行配置。詳細設定と同じ刻み。
//   ★ チェック付きのグループはタイトル帯が高いので、先頭行は 32 から。
//     25 にするとコンボの上端が枠に当たる。
const int kAudioRowTop  = 32;
const int kAudioRowStep = 31;


// 中身の横幅（x=10 に幅 530 のグループが並ぶ = 右端 540）とフッターの高さ
const int kContentWidth = 550;
const int kFooterHeight = 38;

// ★ 縦スクロールバー（約 17px）が出てもグループの右端 540 が隠れない幅。
//   横は出さない設定なので、はみ出るのは 540〜550 の余白だけ。
const int kWindowWidth = 558;

// ウィンドウの高さ（起動時の既定値）。上限は maxContentHeight() が決める。
const int kWindowHeight = 780;
}

encode_setting::encode_setting(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::encode_setting)
{
    ui->setupUi(this);
    setupScrollArea();
    setupLabels();

    settingmap[0] = {"h264_nvenc","0",0,"p1","default","cbr","1pass",1,1};
    settingmap[1] = {"hevc_nvenc","1",1,"p2","hq","vbr","2pass-quarter-res",15,2};
    settingmap[2] = {"av1_nvenc","",2,"p3","ll","cqp","2pass-full-res",30,4};
    settingmap[3] = {"","",3,"p4","ull","cq","",60,8};
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
            const QString ext = encodeSettings.container;
            QString filter = QString("%1 Files (*.%2);;All Files (*.*)")
                                 .arg(ext.toUpper()).arg(ext);
            QFileInfo fileInfo(encodeSettings.encode_path);

            QString fileName = QFileDialog::getSaveFileName(
                this,
                tr("名前を付けて保存"),
                fileInfo.absolutePath(),        // 初期ディレクトリ
                filter
                );

            if (!fileName.isEmpty()) {
                encodeSettings.encode_path = fileName;
                ui->label_filepath->setText(fileName);
                // ★ 別の拡張子で保存されたらコンボをそちらへ合わせる。
                //   muxer は拡張子から決まるので、パスのほうが正。
                syncContainerFromPath();
            }
        }, Qt::QueuedConnection);
    }

    //タイルごとに別ファイルへ出力
    {
        QObject::connect(ui->checkBox_separate, &QCheckBox::toggled, this, [&](bool on) {
            encodeSettings.separate_files = on;
        }, Qt::QueuedConnection);
    }

    //録画フォーマット（コンテナ）
    {
        ui->comboBox_container->addItem("MP4 (.mp4)");
        ui->comboBox_container->addItem("MKV (.mkv)");
        ui->comboBox_container->addItem("MOV (.mov)");
        QObject::connect(ui->comboBox_container, &QComboBox::currentIndexChanged,
                         this, [&](int index) {
            static const char* kExt[] = { "mp4", "mkv", "mov" };
            encodeSettings.container = kExt[qBound(0, index, 2)];
            applyContainerToPath();
            applyContainerCodecLimit();
            rebuildAudioCodecs();
        }, Qt::QueuedConnection);
    }

    //音声
    {
        QObject::connect(ui->audio_groupBox, &QGroupBox::toggled, this, [&](bool on) {
            encodeSettings.audio_enable = on;
        }, Qt::QueuedConnection);

        QObject::connect(ui->comboBox_acodec, &QComboBox::currentIndexChanged,
                         this, [&](int) {
            const QString id = ui->comboBox_acodec->currentData().toString();
            if (!id.isEmpty()) encodeSettings.audio_codec = id;
            applyAudioVisibility();
        }, Qt::QueuedConnection);

        ui->spinBox_again->setKeyboardTracking(false);

        // ★ ビットレートは kbps で扱い、設定へは bps で入れる。
        //   刻みはプラグインに合わせて 16 / ページ 32。
        ui->horizontalSlider_abitrate->setRange(64, 512);
        ui->horizontalSlider_abitrate->setSingleStep(16);
        ui->horizontalSlider_abitrate->setPageStep(32);
        ui->spinBox_abitrate->setKeyboardTracking(false);
        ui->spinBox_abitrate->setSingleStep(16);

        QObject::connect(ui->horizontalSlider_abitrate, &QSlider::valueChanged,
                         this, [&](int kb) {
            encodeSettings.audio_bitrate = kb * 1000;
            syncSpin(ui->horizontalSlider_abitrate, ui->spinBox_abitrate);
        }, Qt::QueuedConnection);
        QObject::connect(ui->spinBox_abitrate, QOverload<int>::of(&QSpinBox::valueChanged),
                         ui->horizontalSlider_abitrate, [this](int v){
            ui->horizontalSlider_abitrate->setValue(v);
        });

        // ★ ゲインは dB。0 で素通し。
        ui->horizontalSlider_again->setRange(-30, 30);
        ui->horizontalSlider_again->setSingleStep(1);
        ui->horizontalSlider_again->setPageStep(3);
        QObject::connect(ui->horizontalSlider_again, &QSlider::valueChanged,
                         this, [&](int db) {
            encodeSettings.audio_gain_db = db;
            syncSpin(ui->horizontalSlider_again, ui->spinBox_again);
        }, Qt::QueuedConnection);
        QObject::connect(ui->spinBox_again, QOverload<int>::of(&QSpinBox::valueChanged),
                         ui->horizontalSlider_again, [this](int v){
            ui->horizontalSlider_again->setValue(v);
        });
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
            encodeSettings.codec = settingmap.value(index).codec;
            combo_index_control2();
            setGPUTable();
            updateBitrateInfo();   // 上限クランプと VRAM が変わる
        }, Qt::QueuedConnection);
    }

    //スプリットエンコード
    {
        QStringList splitencode_items;
        splitencode_items << "Disable" << "Enable" ;
        ui->comboBox_splitencode->addItems(splitencode_items);
        QObject::connect(ui->comboBox_splitencode, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.split_encode_mode = settingmap.value(index).splitencode_items;
        }, Qt::QueuedConnection);
    }

    //タイルエンコードプロファイル
    {
        QStringList tile_items;
        tile_items <<"1 Tile (通常)"<<"2 Tiles (2 × 1)"<<"4 Tiles (2 × 2)"<<"8 Tiles (4 × 2)";
        ui->comboBox_tile->addItems(tile_items);
        QObject::connect(ui->comboBox_tile, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.encode_tile = settingmap.value(index).encode_tile;
            tile_split_exchange();
            updateGpuTileAllocation();
            // ★ 合計基準のときはタイル数で範囲が変わる。
            //   合計値は据え置き = 1 本あたりが変わる、が「合計」の意味。
            applyBitrateRange();

            if(VideoInfo.width*VideoInfo.width_scale/encodeSettings.width_tile>4096||VideoInfo.height*VideoInfo.height_scale/encodeSettings.height_tile>4096){
                hideRow(ui->comboBox_codec, 0, true);
                if(ui->comboBox_codec->currentIndex()==0){
                    ui->comboBox_codec->setCurrentIndex(1);
                }
            }else{
                hideRow(ui->comboBox_codec, 0, false);
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
                encodeSettings.b_frames = settingmap.value(index).B_frame_items;
                updateBitrateInfo();   // リング段数 = VRAM が変わる
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
            encodeSettings.gop_size = settingmap.value(index).gop_items;
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
            encodeSettings.preset = settingmap.value(index).preset_items;
        }, Qt::QueuedConnection);
    }

    //最適化プロファイル（チューニング）
    {
        QStringList profile_items;
        profile_items<< "default (汎用)"<< "hq (高品質)"<< "ll (低遅延)"<< "ull (超低遅延)"<< "lossless (可逆圧縮)";
        ui->comboBox_profile->addItems(profile_items);
        QObject::connect(ui->comboBox_profile, &QComboBox::currentIndexChanged, this, [&](int index) {
            encodeSettings.tune = settingmap.value(index).profile_items;
        }, Qt::QueuedConnection);
    }

    // ★ 入力欄はスライダーと対。本流はスライダー側で、入力欄の値はまず
    //   スライダーへ流してクランプを通す。
    //   setKeyboardTracking(false) は必須（無いと 100 の 1 文字目で弾かれる）。
    {
        QSpinBox* const spins[]   = { ui->spinBox_targetbitrate,
                                      ui->spinBox_maxbitrate,
                                      ui->spinBox_cq };
        QSlider*  const sliders[] = { ui->horizontalSlider_targetbitrate,
                                      ui->horizontalSlider_maxbitrate,
                                      ui->horizontalSlider_cq };
        // ★ ここで繋ぐのは「入力欄 -> スライダー」だけ。
        //   逆向きは各スライダーの valueChanged で syncSpin() を呼ぶ。
        //   範囲のクランプや VBR の連動を通した「結果」を書き戻したいので、
        //   ここでまとめて繋ぐと途中の値が入力欄に出てしまう。
        for (int i = 0; i < 3; i++) {
            spins[i]->setKeyboardTracking(false);
            spins[i]->setRange(sliders[i]->minimum(), sliders[i]->maximum());
            spins[i]->setValue(sliders[i]->value());
            QSlider* sl = sliders[i];
            QObject::connect(spins[i], QOverload<int>::of(&QSpinBox::valueChanged),
                             sl, [sl](int v){ sl->setValue(v); });
        }
    }

    //ビットレート基準
    //   ★ 単位を変えても実際のビットレートは変えない。表示値だけ換算する。
    {
        QStringList basis_items;
        basis_items << "1 ストリームあたり" << "合計（ファイル全体）";
        ui->comboBox_basis->addItems(basis_items);
        QObject::connect(ui->comboBox_basis, &QComboBox::currentIndexChanged, this, [&](int index) {
            const QString next = (index == 1) ? "total" : "stream";
            if (next != encodeSettings.bitrate_basis) {
                const int tiles = qMax(1, encodeSettings.encode_tile);
                if (next == "total") {
                    target_bit_rate *= tiles;
                    max_bit_rate    *= tiles;
                } else {
                    target_bit_rate = qMax(1, target_bit_rate / tiles);
                    max_bit_rate    = qMax(1, max_bit_rate    / tiles);
                }
                encodeSettings.bitrate_basis = next;
            }
            applyBitrateRange();
        }, Qt::QueuedConnection);
    }

    // ターゲットビットレートスライダー
    QObject::connect(ui->horizontalSlider_targetbitrate, &QSlider::valueChanged, this, [&](int value){
        target_bit_rate = value;
        syncSpin(ui->horizontalSlider_targetbitrate, ui->spinBox_targetbitrate);
        updateBitrateInfo();
    }, Qt::QueuedConnection);

    // 最大ビットレートスライダー
    QObject::connect(ui->horizontalSlider_maxbitrate, &QSlider::valueChanged, this, [&](int value){
        max_bit_rate = value;

        // VBR は 目標 <= 最大 に抑える
        if(encodeSettings.rc_mode=="vbr"){
            ui->horizontalSlider_targetbitrate->setMaximum(value);
            if(target_bit_rate > value){
                target_bit_rate = value;
                ui->horizontalSlider_targetbitrate->setValue(value);
            }
        }

        // ★ 目標側も揃えること。setMaximum() は値が収まっていれば
        //   valueChanged を出さないので、上限の変化が入力欄へ届かない。
        syncSpin(ui->horizontalSlider_maxbitrate,    ui->spinBox_maxbitrate);
        syncSpin(ui->horizontalSlider_targetbitrate, ui->spinBox_targetbitrate);
        updateBitrateInfo();
    }, Qt::QueuedConnection);

    //cq
    ui->horizontalSlider_cq->setRange(1, 51);
    QObject::connect(ui->horizontalSlider_cq, &QSlider::valueChanged, this, [&](int value){
        encodeSettings.cq = value;
        syncSpin(ui->horizontalSlider_cq, ui->spinBox_cq);
    }, Qt::QueuedConnection);


    //レート制御
    //   ★ 並びと表記は tbve-rt(OBS) 側に合わせてある。
    //     CBR : 目標のみ（最大は目標と同じ値を入れる）
    //     VBR : 目標 + 最大
    //     CQP : 固定量子化。ビットレートは一切使わない
    //     CQ  : 品質指定VBR。ビットレートは指定せず、最大だけ上限として使う
    {
        QStringList rc_items;
        rc_items << "CBR" << "VBR" << "CQP" << "CQ (品質指定VBR)";
        ui->comboBox_rc->addItems(rc_items);
        QObject::connect(ui->comboBox_rc, &QComboBox::currentIndexChanged,this, [&](int index) {
            encodeSettings.rc_mode = settingmap.value(index).rc_items;
            applyRcVisibility();
            applyBitrateRange();
        }, Qt::QueuedConnection);
    }

    //エンコードパス
    {
        QStringList pass_items;
        pass_items << "1 pass" << "2 pass quarter" << "2 pass full" ;
        ui->comboBox_encodepass->addItems(pass_items);
        QObject::connect(ui->comboBox_encodepass, &QComboBox::currentIndexChanged,this, [&](int index) {
            encodeSettings.pass_mode = settingmap.value(index).pass_items;
        }, Qt::QueuedConnection);
    }

    //エンコードスタート
    QObject::connect(ui->encodeStart_pushbutton, &QPushButton::clicked, this, [&]() {
        // ★ 保存は常に 1 ストリームあたり。合計基準ならタイル数で割り戻す。
        const int unit = bitrate_unit();
        qint64 tgt = (qint64)target_bit_rate / unit;
        qint64 mx  = (encodeSettings.rc_mode=="cbr") ? tgt : ((qint64)max_bit_rate / unit);
        if(tgt < 1) tgt = 1;
        if(mx  < 1) mx  = 1;
        tgt *= 1000*1000;
        mx  *= 1000*1000;

        // ★ target と max は別々に通すこと（旧実装はここを取り違えていた）。
        const qint64 cap = codec_max_bitrate(encodeSettings.codec);
        encodeSettings.target_bit_rate = (int)qMin(tgt, cap);
        encodeSettings.max_bit_rate    = (int)qMin(mx,  cap);

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
        ui->audio_groupBox->setEnabled(false);
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
        ui->audio_groupBox->setEnabled(VideoInfo.audio);
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
        //   ★ コンテナはパスの拡張子から決める（muxer を決めるのは拡張子）。
        ui->label_filepath->setText(encodeSettings.encode_path);
        ui->checkBox_separate->setChecked(encodeSettings.separate_files);
        syncContainerFromPath();          // この中で rebuildAudioCodecs() も走る

        ui->horizontalSlider_abitrate->setValue(
            qBound(64, encodeSettings.audio_bitrate / 1000, 512));
        syncSpin(ui->horizontalSlider_abitrate, ui->spinBox_abitrate);

        ui->horizontalSlider_again->setValue(
            qBound(-30, (int)qRound(encodeSettings.audio_gain_db), 30));
        syncSpin(ui->horizontalSlider_again, ui->spinBox_again);
        ui->audio_groupBox->setChecked(encodeSettings.audio_enable);
        applyAudioVisibility();

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

        // ★ 範囲がタイル数と基準に依存するので、tile を確定させてから。
        ui->comboBox_basis->setCurrentIndex(combo_index[9]);
        emitIndexChanged(ui->comboBox_basis);
        applyBitrateRange();

        ui->horizontalSlider_cq->setValue(encodeSettings.cq);
        syncSpin(ui->horizontalSlider_cq, ui->spinBox_cq);
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
    // ★ foundIndex() は見つからないと -1 を返す。そのまま
    //   setCurrentIndex(-1) すると選択が外れ、ハンドラにも -1 が渡る。
    //   既定（先頭）へ倒す。
    const auto fi = [&](const char* key, const QString& v){
        return qMax(0, foundIndex(key, v));
    };

    combo_index[0] = fi("codec", "h264_nvenc");
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
        combo_index[0] = fi("codec", encodeSettings.codec);
    }
    else
    {
        // primaryがAV1非対応、または見つからない場合
        if (encodeSettings.codec == "av1_nvenc")
            combo_index[0] = fi("codec", "hevc_nvenc");
        else
            combo_index[0] = fi("codec", encodeSettings.codec);
    }
    combo_index[1]= fi("split_encode_mode",encodeSettings.split_encode_mode);
    combo_index[2]= fi("b_frames",QString::number(encodeSettings.b_frames));
    combo_index[3]= fi("preset",encodeSettings.preset);
    combo_index[4]= fi("tune",encodeSettings.tune);
    combo_index[5]= fi("rc_mode",encodeSettings.rc_mode);
    combo_index[6]= fi("pass_mode",encodeSettings.pass_mode);
    combo_index[7]= fi("gop_size",QString::number(encodeSettings.gop_size));
    combo_index[8]= fi("encode_tile",QString::number(encodeSettings.encode_tile));
    combo_index[9]=(encodeSettings.bitrate_basis=="total") ? 1 : 0;

    // ★ 保存値は常に 1 本あたり。表示単位へ戻す。
    const int unit = bitrate_unit();
    target_bit_rate = encodeSettings.target_bit_rate/1000000*unit;
    max_bit_rate    = encodeSettings.max_bit_rate/1000000*unit;
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

// ★ ラベルは日本語、ffmpeg のフラグはツールチップへ。
//   ラベルと入力側の両方に付けるので、どちらに触れても出る。
void encode_setting::setupLabels()
{
    const struct { QLabel* label; QWidget* field; const char* flag; } rows[] = {
        { ui->label_3,             ui->comboBox_codec,                  "-c:v"                },
        { ui->label_4,             ui->comboBox_framerate,              "-r"                  },
        { ui->label_6,             ui->comboBox_splitencode,            "-split_encode_mode"  },
        { ui->label_8,             ui->comboBox_preset,                 "-preset"             },
        { ui->label_9,             ui->comboBox_profile,                "-tune"               },
        { ui->label_10,            ui->comboBox_gop,                    "-g"                  },
        { ui->label_13,            ui->comboBox_b_frame,                "-bf"                 },
        { ui->label_11,            ui->comboBox_rc,                     "-rc"                 },
        { ui->label_12,            ui->comboBox_encodepass,             "-multipass"          },
        { ui->label_targetbitrate, ui->horizontalSlider_targetbitrate,  "-b:v"                },
        { ui->label_maxbitrate,    ui->horizontalSlider_maxbitrate,     "-maxrate"            },
    };

    for (const auto& r : rows) {
        r.label->setToolTip(r.flag);
        r.field->setToolTip(r.flag);
    }

    // スライダーと対になっている入力欄にも同じものを付ける
    ui->spinBox_targetbitrate->setToolTip("-b:v");
    ui->spinBox_maxbitrate->setToolTip("-maxrate");

    // フラグが無いもの（この実装固有の設定）
    ui->comboBox_container->setToolTip(tr("出力ファイルの拡張子で muxer が決まります"));
    ui->comboBox_tile->setToolTip(tr("キャンバスを分割して複数の NVENC で並列にエンコードします"));
    ui->label_abitrate->setToolTip("-b:a");
    ui->horizontalSlider_abitrate->setToolTip("-b:a");
    ui->spinBox_abitrate->setToolTip("-b:a");
    ui->horizontalSlider_again->setToolTip(tr("0 で素通し。float のうちに掛けます"));
    ui->spinBox_again->setToolTip(tr("0 で素通し。float のうちに掛けます"));
    ui->checkBox_separate->setToolTip(
        tr("out.mp4 -> out.tile0.mp4 / out.tile1.mp4 ...  音声はタイル 0 のファイルに入ります"));
    ui->comboBox_basis->setToolTip(
        tr("ビットレートを 1 ストリームあたりで指定するか、ファイル全体の合計で指定するか"));
}

// ★ レート制御で使わない行を隠し、見えている行を上から詰める。
//   無効化ではなく非表示にする（docs/encode-settings.md）。
void encode_setting::applyRcVisibility()
{
    const QString rc = encodeSettings.rc_mode;
    const bool useTarget = (rc=="cbr" || rc=="vbr");
    const bool useMax    = (rc=="vbr" || rc=="cq");     // CQ では上限としてだけ使う
    const bool useQual   = (rc=="cq"  || rc=="cqp");
    const bool useBasis  = (useTarget || useMax);       // ビットレートを使うときだけ

    ui->label_targetbitrate->setVisible(useTarget);
    ui->horizontalSlider_targetbitrate->setVisible(useTarget);
    ui->spinBox_targetbitrate->setVisible(useTarget);

    ui->label_maxbitrate->setVisible(useMax);
    ui->horizontalSlider_maxbitrate->setVisible(useMax);
    ui->spinBox_maxbitrate->setVisible(useMax);

    ui->label_cq->setVisible(useQual);
    ui->horizontalSlider_cq->setVisible(useQual);
    ui->spinBox_cq->setVisible(useQual);

    ui->label_basis->setVisible(useBasis);
    ui->comboBox_basis->setVisible(useBasis);

    // ★ 品質の意味が違うのでラベルを入れ替える。範囲はどちらも 1〜51。
    //   0 は NVENC が受け付けずエンコーダの生成に失敗する。
    ui->horizontalSlider_cq->blockSignals(true);
    // ★ CQ と CQP で意味が違うので文言ごと入れ替える。
    //   ffmpeg のフラグはツールチップに回してある（setupLabels）。
    ui->label_cq->setText(rc=="cqp" ? tr("量子化 (QP)") : tr("目標品質 (CQ)"));
    ui->label_cq->setToolTip(rc=="cqp" ? "-qp" : "-cq");
    ui->spinBox_cq->setToolTip(ui->label_cq->toolTip());
    ui->horizontalSlider_cq->setToolTip(ui->label_cq->toolTip());
    ui->horizontalSlider_cq->setRange(1, 51);
    encodeSettings.cq = qBound(ui->horizontalSlider_cq->minimum(),
                               encodeSettings.cq,
                               ui->horizontalSlider_cq->maximum());
    ui->horizontalSlider_cq->setValue(encodeSettings.cq);
    ui->horizontalSlider_cq->blockSignals(false);
    syncSpin(ui->horizontalSlider_cq, ui->spinBox_cq);

    // ---- 見えている行を上から詰める ----
    int slot = 0;
    const int top = useBasis ? kRateRowTop : kBasisRowTop;
    const auto place = [&](bool on, QWidget* lab, QWidget* sld, QWidget* val){
        if (!on) return;
        const int y = top + slot * kRateRowStep;
        lab->move(lab->x(), y + 1);
        sld->move(sld->x(), y);
        val->move(val->x(), y);      // 入力欄はスライダーと同じ高さ
        slot++;
    };
    place(useTarget, ui->label_targetbitrate, ui->horizontalSlider_targetbitrate,
                     ui->spinBox_targetbitrate);
    place(useMax,    ui->label_maxbitrate,    ui->horizontalSlider_maxbitrate,
                     ui->spinBox_maxbitrate);
    place(useQual,   ui->label_cq,            ui->horizontalSlider_cq,
                     ui->spinBox_cq);

    // ★ 換算行はビットレートを使うモードでだけ出す（CQP では 1 行も出ない）。
    rate_rows_   = slot;
    use_basis_   = useBasis;
    info_target_ = useTarget;
    info_max_    = useMax;

    // 消えた行のぶんだけグループを縮め、下の要素を引き上げる。
    relayoutContent();
}

// ★ 上半分をスクロール領域へ移す。座標はそのまま使える。
//   setParent は widget を隠すので show() を忘れないこと。
void encode_setting::setupScrollArea()
{
    scrollArea_ = new QScrollArea(this);
    scrollArea_->setFrameShape(QFrame::NoFrame);
    scrollArea_->setWidgetResizable(false);
    scrollArea_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    // ★ 横は出さない。中身の右端 511 のうち、見えなくなるのは
    //   グループの外（501〜511）の余白だけなので実害が無い。
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea_->setBackgroundRole(QPalette::Window);
    scrollArea_->viewport()->setBackgroundRole(QPalette::Window);

    content_ = new QWidget;
    content_->setBackgroundRole(QPalette::Window);
    content_->setGeometry(0, 0, kContentWidth, height());

    QWidget* const move_in[] = {
        ui->filepath_groupBox,
        ui->normalsetting_groupBox,
        ui->advancesetting_groupBox,
        ui->audio_groupBox,
        ui->label_GPUTable,
        ui->tableWidget_gpu
    };
    for (QWidget* w : move_in) {
        w->setParent(content_);
        w->show();
    }

    scrollArea_->setWidget(content_);
    scrollArea_->show();

    // ★ 大きさを決めるのはここ 1 回だけ。以降は項目が増減しても触らない。
    //   小さくされてもフッターが潰れない程度の下限は入れておく。
    // ★ 横は固定。中身が固定座標なので、広げても右に余白が増えるだけ。
    setFixedWidth(kWindowWidth);
    setMinimumHeight(240);
    resize(kWindowWidth, kWindowHeight);
    layoutFooter();
}

// ★ ボタンと進捗バーは常にウィンドウ下端。スクロールの外。
void encode_setting::layoutFooter()
{
    if (!scrollArea_) return;

    const int footerTop = qMax(0, height() - kFooterHeight);
    scrollArea_->setGeometry(0, 0, width(), footerTop);

    const int by = footerTop + 7;
    ui->encodeStart_pushbutton->move(10, by);
    ui->encodecancel_pushButton->move(93, by);
    ui->encode_progressBar->setGeometry(180, by, qMax(120, width() - 194), 23);
}

// ★ 行数から詳細設定グループの高さを決め、下流を積み直す。
void encode_setting::relayoutContent()
{
    if (!content_) return;

    // ---- スライダー行の下端 ----
    const int top = use_basis_ ? kRateRowTop : kBasisRowTop;
    int bottom;
    if (rate_rows_ > 0)     bottom = top + (rate_rows_ - 1) * kRateRowStep + kRowHeight;
    else if (use_basis_)    bottom = kBasisRowTop + kRowHeight;
    else                    bottom = kBasisRowTop;

    // ---- 情報行をその下にぶら下げる ----
    int y = bottom + kInfoGap;
    const auto placeInfo = [&](QWidget* w, bool on){
        w->setVisible(on);
        if (!on) return;
        w->move(w->x(), y);
        y += kInfoStep;
        bottom = y - kInfoStep + kInfoHeight;
    };
    placeInfo(ui->label_rate_info1, info_target_);
    placeInfo(ui->label_rate_info2, info_max_);
    placeInfo(ui->label_vram_info,  true);          // VRAM は常に出す

    const QRect r = ui->advancesetting_groupBox->geometry();
    ui->advancesetting_groupBox->setGeometry(r.x(), r.y(), r.width(), bottom + kGroupPad);

    const int advEnd = r.y() + bottom + kGroupPad;

    // GPU 表 -> 音声グループ の順に下へ積む
    ui->label_GPUTable->move(ui->label_GPUTable->x(), advEnd + 8);
    ui->tableWidget_gpu->move(ui->tableWidget_gpu->x(), advEnd + 29);
    const int tblEnd = ui->tableWidget_gpu->y() + ui->tableWidget_gpu->height();

    const QRect ar = ui->audio_groupBox->geometry();
    ui->audio_groupBox->setGeometry(ar.x(), tblEnd + 6, ar.width(), audio_h_);

    // ★ 中身の高さだけを更新する。ウィンドウには触らない。
    //   ビューポートより高ければ縦スクロールが出て、低ければ出ない。
    //   どちらでも窓は動かない。
    const int ch = ui->audio_groupBox->y() + ui->audio_groupBox->height() + kGroupPad;
    content_->setFixedSize(kContentWidth, ch);

    // ★ 全部収まったらそれ以上は伸ばせないようにする。
    //   いまのモードの高さを上限にすると、モードを切り替えるたびに
    //   ウィンドウが縮んで落ち着かない。いちばん背の高いモードで決める。
    setMaximumHeight(maxContentHeight() + kFooterHeight);
}

// ★ いちばん項目が多い状態（VBR + 非可逆音声）の中身の高さ。
//   いまのモードを上限にすると切り替えのたびに窓が縮むので、最大で決める。
int encode_setting::maxContentHeight() const
{
    // スライダー 2 行
    int bottom = kRateRowTop + kRateRowStep + kRowHeight;

    // 情報行 3 本（目標 / 最大 / VRAM）
    int y = bottom + kInfoGap;
    for (int i = 0; i < 3; i++) { bottom = y + kInfoHeight; y += kInfoStep; }

    const int advEnd = ui->advancesetting_groupBox->y() + bottom + kGroupPad;
    const int tblEnd = advEnd + 29 + ui->tableWidget_gpu->height();

    // 音声 3 行
    const int audioH = kAudioRowTop + 2 * kAudioRowStep + kRowHeight + kGroupPad;

    return tblEnd + 6 + audioH + kGroupPad;
}

void encode_setting::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    layoutFooter();
}

// ★ EncodeSettings に入るのは常に 1 ストリームあたり。合計で見せている
//   間はこの数で割って保存する（docs/encode-settings.md）。
void encode_setting::syncSpin(QSlider* slider, QSpinBox* spin)
{
    if (!slider || !spin) return;
    QSignalBlocker b(spin);               // 書き戻してスライダーを動かさない
    spin->setRange(slider->minimum(), slider->maximum());
    spin->setValue(slider->value());
}

int encode_setting::bitrate_unit() const
{
    return (encodeSettings.bitrate_basis=="total")
               ? qMax(1, encodeSettings.encode_tile) : 1;
}

// ★ 1 本あたりが常に 1〜1000 Mbps に収まるよう範囲を決め直す。
void encode_setting::applyBitrateRange()
{
    const int unit = bitrate_unit();
    const int hi   = 1000 * unit;

    ui->horizontalSlider_maxbitrate->blockSignals(true);
    ui->horizontalSlider_targetbitrate->blockSignals(true);

    ui->horizontalSlider_maxbitrate->setRange(unit, hi);
    max_bit_rate = qBound(unit, max_bit_rate, hi);
    ui->horizontalSlider_maxbitrate->setValue(max_bit_rate);

    // VBR のときだけ 目標 <= 最大 に縛る
    const int thi = (encodeSettings.rc_mode=="vbr") ? max_bit_rate : hi;
    ui->horizontalSlider_targetbitrate->setRange(unit, thi);
    target_bit_rate = qBound(unit, target_bit_rate, thi);
    ui->horizontalSlider_targetbitrate->setValue(target_bit_rate);

    ui->horizontalSlider_maxbitrate->blockSignals(false);
    ui->horizontalSlider_targetbitrate->blockSignals(false);

    // ★ blockSignals で止めたぶん、入力欄には届いていない。ここで揃える。
    syncSpin(ui->horizontalSlider_targetbitrate, ui->spinBox_targetbitrate);
    syncSpin(ui->horizontalSlider_maxbitrate,    ui->spinBox_maxbitrate);

    updateBitrateInfo();
}

// ★ 入力欄の単位と、タイルごとの換算・VRAM の表示。
void encode_setting::updateBitrateInfo()
{
    const int  tiles = qMax(1, encodeSettings.encode_tile);
    const bool total = (encodeSettings.bitrate_basis=="total");
    const QString unit = (total || tiles==1) ? " Mbps" : " Mbps/本";

    // ★ 単位は入力欄のサフィックスで見せる（値そのものはスライダーが持つ）
    ui->spinBox_targetbitrate->setSuffix(unit);
    ui->spinBox_maxbitrate->setSuffix(unit);

    // ★ 換算は常に 1 本あたりで出す（NVENC が受け取るのがその値）。
    //   CQ は目標を使わないので、見せるのは上限のほう。
    const int unitDiv = bitrate_unit();
    const auto line = [&](const QString& name, int shown){
        const int ps = qMax(1, shown / unitDiv);
        return (tiles <= 1)
                   ? tr("%1 %2 Mbps").arg(name).arg(ps)
                   : tr("%1 %2 Mbps/本 × %3 タイル = 合計 %4 Mbps")
                         .arg(name).arg(ps).arg(tiles).arg((qint64)ps * tiles);
    };

    if (info_target_) ui->label_rate_info1->setText(line(tr("目標"), target_bit_rate));
    if (info_max_)    ui->label_rate_info2->setText(line(tr("最大"), max_bit_rate));

    // ★ 式は __global__.h（save_encode と共有。表示と実体をずらさない）。
    //   B-frame はエンコーダ上限でクランプされるので同じ値を通す。
    const int bf   = qBound(0, qMin(encodeSettings.b_frames,
                                    nvenc_max_bframes(encodeSettings.codec)), 7);
    const int ring = encode_ring_size(bf);
    const int tw   = VideoInfo.width  * VideoInfo.width_scale
                         / qMax(1, encodeSettings.width_tile);
    const int th   = VideoInfo.height * VideoInfo.height_scale
                         / qMax(1, encodeSettings.height_tile);
    const int mb   = encode_pool_mb(tw, th, ring, tiles);

    ui->label_vram_info->setText(
        tr("想定使用 VRAM %1 MB   (%2 x %3 x %4 面 x %5 タイル)")
            .arg(mb).arg(tw).arg(th).arg(ring + 4).arg(tiles));
}

void encode_setting::applyContainerToPath()
{
    if (path_sync_) return;

    const QFileInfo fi(encodeSettings.encode_path);
    const QString np = fi.absolutePath() + "/" + fi.completeBaseName()
                       + "." + encodeSettings.container;
    if (np == encodeSettings.encode_path) return;

    encodeSettings.encode_path = np;
    ui->label_filepath->setText(np);
}

// ★ 知らない拡張子のときはコンボを正として付け直す。
//   中途半端な拡張子だと muxer が決まらず avformat_alloc... が失敗する。
void encode_setting::syncContainerFromPath()
{
    const QString ext = QFileInfo(encodeSettings.encode_path).suffix().toLower();

    int idx = -1;
    if      (ext == "mp4") idx = 0;
    else if (ext == "mkv") idx = 1;
    else if (ext == "mov") idx = 2;

    if (idx < 0) {              // 知らない拡張子。コンボに合わせて付け直す
        applyContainerToPath();
        applyContainerCodecLimit();
        rebuildAudioCodecs();
        return;
    }

    path_sync_ = true;          // ここからの currentIndexChanged で書き戻させない
    encodeSettings.container = ext;
    ui->comboBox_container->setCurrentIndex(idx);
    path_sync_ = false;

    applyContainerCodecLimit();
    rebuildAudioCodecs();
}

// ★ コンテナに入る音声コーデックだけを並べ直す。消えるものを選んで
//   いたら AAC へ逃がす（docs/encode-settings.md の対応表）。
void encode_setting::rebuildAudioCodecs()
{
    const QString c = encodeSettings.container;
    const bool pcm_ok  = (c != "mp4");   // MP4 の PCM は ipcm で再生環境を選ぶ
    const bool flac_ok = (c != "mov");   // MOV は FLAC を格納できない

    // 顔ぶれが変わらないなら触らない（選択が飛ぶのを防ぐ）
    const int key = (pcm_ok ? 1 : 0) | (flac_ok ? 2 : 0);
    if (acodec_key_ == key && ui->comboBox_acodec->count() > 0) return;
    acodec_key_ = key;

    const QString cur = encodeSettings.audio_codec;

    QSignalBlocker block(ui->comboBox_acodec);
    ui->comboBox_acodec->clear();
    ui->comboBox_acodec->addItem(tr("AAC (非可逆)"), "aac");
    if (flac_ok) ui->comboBox_acodec->addItem(tr("FLAC (可逆圧縮)"), "flac");
    if (pcm_ok) {
        ui->comboBox_acodec->addItem(tr("PCM 16bit (非圧縮)"),       "pcm_s16le");
        ui->comboBox_acodec->addItem(tr("PCM 24bit (非圧縮)"),       "pcm_s24le");
        ui->comboBox_acodec->addItem(tr("PCM 32bit float (非圧縮)"), "pcm_f32le");
    }

    const int i = ui->comboBox_acodec->findData(cur);
    ui->comboBox_acodec->setCurrentIndex(i >= 0 ? i : 0);
    encodeSettings.audio_codec = ui->comboBox_acodec->currentData().toString();

    applyAudioVisibility();
}

// ★ 可逆はビットレートを指定できないので、行ごと消して高さも詰める。
void encode_setting::applyAudioVisibility()
{
    const QString a = encodeSettings.audio_codec;
    const bool lossless = a.startsWith("pcm_") || a == "flac";

    // ★ 見えている行を上から詰める（詳細設定のスライダー行と同じ要領）
    int slot = 0;
    const auto place = [&](bool on, QWidget* lab, QWidget* sld, QWidget* val){
        lab->setVisible(on);
        if (sld) sld->setVisible(on);
        if (val) val->setVisible(on);
        if (!on) return;
        const int y = kAudioRowTop + slot * kAudioRowStep;
        lab->move(lab->x(), y + 1);
        if (sld) sld->move(sld->x(), y);
        if (val) val->move(val->x(), y);
        slot++;
    };
    place(true,      ui->label_acodec,   nullptr,                        ui->comboBox_acodec);
    place(!lossless, ui->label_abitrate, ui->horizontalSlider_abitrate,  ui->spinBox_abitrate);
    place(true,      ui->label_again,    ui->horizontalSlider_again,     ui->spinBox_again);

    audio_h_ = kAudioRowTop + (slot - 1) * kAudioRowStep + kRowHeight + kGroupPad;

    relayoutContent();
}

// ★ mov muxer は AV1 を素直に受けないので項目ごと隠す。
//   版によっては通るので、実機で確認が取れたら外してよい。
void encode_setting::applyContainerCodecLimit()
{
    if (ui->comboBox_codec->count() <= 2) return;   // そもそも AV1 が無い

    const bool mov = (encodeSettings.container == "mov");
    hideRow(ui->comboBox_codec, 2, mov);

    if (mov && ui->comboBox_codec->currentIndex() == 2)
        ui->comboBox_codec->setCurrentIndex(1);              // hevc へ
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
    ui->audio_groupBox->setEnabled(VideoInfo.audio);
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
    updateBitrateInfo();   // 解像度が確定したので VRAM を出し直す

    // ★ 音声を持たない素材ではグループごと触れなくする
    ui->audio_groupBox->setEnabled(VideoInfo.audio);
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
        combo_index_control(ui->comboBox_b_frame,0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,1,7,true,true);
        combo_index_control(ui->comboBox_profile,0,4,false,false);
        combo_index_control(ui->comboBox_profile,0,1,true,true);
        combo_index_control(ui->comboBox_profile,4,4,true,true);
    }else if(encodeSettings.encode_tile>1){
        combo_index_control(ui->comboBox_b_frame,0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,1,7,true,true);
        combo_index_control(ui->comboBox_profile,0,4,false,false);
        combo_index_control(ui->comboBox_profile,4,4,true,true);
        //combo_index_control(ui->comboBox_splitencode,1,1,true,true);
    }else if(encodeSettings.codec == "h264_nvenc"){
        combo_index_control(ui->comboBox_b_frame,0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,5,7,true,true);
        combo_index_control(ui->comboBox_profile,0,4,false,false);
        //combo_index_control(ui->comboBox_splitencode,1,1,false,false);
    }else if(encodeSettings.codec == "hevc_nvenc"){
        combo_index_control(ui->comboBox_b_frame,0,7,false,false);
        combo_index_control(ui->comboBox_b_frame,6,7,true,true);
        combo_index_control(ui->comboBox_profile,0,4,false,false);
        //combo_index_control(ui->comboBox_splitencode,1,1,false,false);
    }else if(encodeSettings.codec == "av1_nvenc"){
        combo_index_control(ui->comboBox_b_frame,0,7,false,false);
        combo_index_control(ui->comboBox_profile,0,4,false,false);
        combo_index_control(ui->comboBox_profile,4,4,true,true);
        //combo_index_control(ui->comboBox_splitencode,1,1,false,false);
    }
}

//特定のコンボボックスを非表示にする
void encode_setting::hideRow(QComboBox* box, int row, bool hide)
{
    if (!box || row < 0 || row >= box->count()) return;
    if (QListView* v = qobject_cast<QListView*>(box->view()))
        v->setRowHidden(row, hide);
}

void encode_setting::combo_index_control(QComboBox* comboBox, int ini_index, int target_index, bool flag,bool index_change_flag) {
    for (int i = ini_index; i <= target_index; i++) {
        hideRow(comboBox, i, flag);
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
        hideRow(ui->comboBox_codec, 0, true);

        if(encodeSettings.codec == "h264_nvenc"||encodeSettings.codec == "hevc_nvenc"){
            ui->comboBox_codec->setCurrentIndex(1);
        }else{
            ui->comboBox_codec->setCurrentIndex(2);
        }
    }else{
        hideRow(ui->comboBox_codec, 0, false);
    }

    //タイル数チェック
    hideRow(ui->comboBox_tile, 0, false);
    hideRow(ui->comboBox_tile, 1, false);
    hideRow(ui->comboBox_tile, 2, false);
    hideRow(ui->comboBox_tile, 3, false);
    //2×2
    if(VideoInfo.width*VideoInfo.width_scale/2>8192||VideoInfo.height*VideoInfo.height_scale/2>8192){
        hideRow(ui->comboBox_tile, 0, true);
        hideRow(ui->comboBox_tile, 1, true);
        hideRow(ui->comboBox_tile, 2, true);
        if(ui->comboBox_tile->currentIndex()<3){
            ui->comboBox_tile->setCurrentIndex(3);
        }
    }

    //2×1
    if(VideoInfo.width*VideoInfo.width_scale/2>8192||VideoInfo.height*VideoInfo.height_scale>8192){
        hideRow(ui->comboBox_tile, 0, true);
        hideRow(ui->comboBox_tile, 1, true);
        if(ui->comboBox_tile->currentIndex()<2){
            ui->comboBox_tile->setCurrentIndex(2);
        }
    }

    //1×1
    if(VideoInfo.width*VideoInfo.width_scale>8192||VideoInfo.height*VideoInfo.height_scale>8192){
        hideRow(ui->comboBox_tile, 0, true);
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
    fitTableHeight();

    // GPUが1枚しかない場合はグレーアウト
    if(g_GPUInfo.size() <= 1){
        multiGPUtabel_UI_Control(false);
    }
}

// ★ 行数ぶんの高さに詰める。表自身がスクロールすると外側のスクロール
//   領域と入れ子になり、ホイールが表に吸われる。
void encode_setting::fitTableHeight()
{
    QTableWidget* tbl = ui->tableWidget_gpu;

    int h = tbl->horizontalHeader()->height() + 2 * tbl->frameWidth();
    for (int r = 0; r < tbl->rowCount(); r++)
        h += tbl->rowHeight(r);

    const QRect g = tbl->geometry();
    tbl->setGeometry(g.x(), g.y(), g.width(), qMax(h, 60));

    relayoutContent();   // 下の音声グループを引き上げる
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
