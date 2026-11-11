#include "src/qmlui/encodecontroller.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QFile>
#include <QRegularExpression>
#include <QDebug>

namespace {
QVariantMap opt(const QString& text, const QVariant& value)
{
    QVariantMap m;
    m["text"]  = text;
    m["value"] = value;
    return m;
}
bool av1Capable(const GPUInfo& g) { return g.CC_major > 8 || (g.CC_major == 8 && g.CC_minor >= 9); }
}

EncodeController::EncodeController(QObject* parent)
    : QObject(parent)
{
    // 使用率・メモリの表示更新（info_thread が 1 秒ごとに g_GPUInfo を更新している）
    statsTimer_.setInterval(1000);
    connect(&statsTimer_, &QTimer::timeout, this, &EncodeController::statsChanged);

    // B フレームは Turing（CC 7.5）以降
    if (!g_GPUInfo.empty()) {
        const GPUInfo& g = g_GPUInfo[0];
        bframe_capable_ = (g.CC_major > 7 || (g.CC_major == 7 && g.CC_minor >= 5));
    }
    loadFromSettings();
    normalize();
}

// ==============================================================
// 初期化（旧 iniFile_UI_Control）
// ==============================================================
void EncodeController::loadFromSettings()
{
    EncodeSettings& s = es();

    // 知らない値は既定へ倒す（旧 foundIndex が -1 のとき先頭にしていたのと同じ）
    auto oneOf = [](const QString& v, const QStringList& list, const QString& def) {
        return list.contains(v) ? v : def;
    };
    s.codec     = oneOf(s.codec, {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}, "h264_nvenc");
    if (s.codec == "av1_nvenc" && !primaryAv1()) s.codec = "hevc_nvenc";
    s.split_encode_mode = oneOf(s.split_encode_mode, {"0", "1"}, "0");
    s.preset    = oneOf(s.preset, {"p1","p2","p3","p4","p5","p6","p7"}, "p1");
    s.tune      = oneOf(s.tune, {"default","hq","ll","ull","lossless"}, "default");
    s.rc_mode   = oneOf(s.rc_mode, {"cbr","vbr","cqp","cq"}, "cbr");
    s.pass_mode = oneOf(s.pass_mode, {"1pass","2pass-quarter-res","2pass-full-res"}, "1pass");
    s.bitrate_basis = (s.bitrate_basis == "total") ? "total" : "stream";
    s.container = oneOf(s.container, {"mp4","mkv","mov"}, "mp4");
    if (!QList<int>{1,15,30,60,120,250,300}.contains(s.gop_size)) s.gop_size = 1;
    if (!QList<int>{1,2,4,8}.contains(s.encode_tile)) s.encode_tile = 1;
    s.b_frames = std::clamp(s.b_frames, 0, 7);
    if (!bframe_capable_) s.b_frames = 0;
    s.cq = std::clamp(s.cq, 1, 51);
    tileSplit(s.encode_tile, s.width_tile, s.height_tile);

    // ★ 保存値は常に 1 本あたり。表示単位へ戻す。
    target_ = s.target_bit_rate / 1000000 * unit();
    max_    = s.max_bit_rate    / 1000000 * unit();

    // コンテナはパスの拡張子から決める（muxer を決めるのは拡張子）
    syncContainerFromPath();
}

// ==============================================================
// 制約（旧 tile_index_control / combo_index_control2 / applyContainerCodecLimit）
// ==============================================================
bool EncodeController::primaryAv1() const
{
    for (const auto& g : g_GPUInfo)
        if (g.openglEnable) return av1Capable(g);
    return false;
}

void EncodeController::tileSplit(int tiles, int& wt, int& ht)
{
    if (tiles == 2)      { wt = 2; ht = 1; }
    else if (tiles == 4) { wt = 2; ht = 2; }
    else if (tiles == 8) { wt = 4; ht = 2; }
    else                 { wt = 1; ht = 1; }
}

// NVENC の 1 本あたりの上限は 8192（H.264 は 4096）
bool EncodeController::tileAllowed(int tiles) const
{
    const int W = canvasW(), H = canvasH();
    switch (tiles) {
    case 1:  return !(W > 8192 || H > 8192);
    case 2:  return !(W / 2 > 8192 || H > 8192);
    case 4:  return !(W / 2 > 8192 || H / 2 > 8192);
    default: return true;
    }
}

bool EncodeController::codecAllowed(const QString& c) const
{
    if (c == "h264_nvenc") {
        int wt = 1, ht = 1;
        tileSplit(es().encode_tile, wt, ht);
        return !(canvasW() / wt > 4096 || canvasH() / ht > 4096);
    }
    if (c == "av1_nvenc")
        return primaryAv1() && es().container != "mov";   // mov muxer は AV1 を素直に受けない
    return true;
}

bool EncodeController::bframeAllowed(int bf) const
{
    if (!bframe_capable_) return bf == 0;
    const EncodeSettings& s = es();
    if (s.gop_size == 1)        return bf == 0;
    if (s.encode_tile > 1)      return bf == 0;
    if (s.codec == "h264_nvenc") return bf <= 4;
    if (s.codec == "hevc_nvenc") return bf <= 5;
    return true;                                  // av1
}

bool EncodeController::tuneAllowed(const QString& t) const
{
    const EncodeSettings& s = es();
    if (s.gop_size == 1)        return t == "ll" || t == "ull";
    if (s.encode_tile > 1)      return t != "lossless";
    if (s.codec == "av1_nvenc") return t != "lossless";
    return true;
}

// ★ 選べなくなった値を選べる値へ寄せる。順序に意味がある：
//   タイル → (分割数) → コーデック → B フレーム / チューニング → 音声 → GPU → レート
void EncodeController::normalize()
{
    EncodeSettings& s = es();

    // タイル：選べなければ、それ以上で最初に選べるもの（旧実装と同じ寄せ方）
    if (!tileAllowed(s.encode_tile)) {
        int pick = 8;
        for (int t : {1, 2, 4, 8}) if (t >= s.encode_tile && tileAllowed(t)) { pick = t; break; }
        s.encode_tile = pick;
    }
    tileSplit(s.encode_tile, s.width_tile, s.height_tile);

    // コーデック：H.264 / AV1 が使えないときは HEVC へ
    if (!codecAllowed(s.codec)) s.codec = "hevc_nvenc";

    // B フレーム：使える範囲の上限まで下げる
    while (s.b_frames > 0 && !bframeAllowed(s.b_frames)) s.b_frames--;

    // チューニング：default/hq は ll へ、lossless は ull へ
    if (!tuneAllowed(s.tune))
        s.tune = (s.tune == "lossless") ? "ull" : "ll";
    if (!tuneAllowed(s.tune)) s.tune = "ull";

    // 音声コーデック：コンテナに入らないものは AAC へ
    {
        const bool pcm_ok  = (s.container != "mp4");
        const bool flac_ok = (s.container != "mov");
        const QString a = s.audio_codec;
        if ((a.startsWith("pcm_") && !pcm_ok) || (a == "flac" && !flac_ok) ||
            !(a == "aac" || a == "flac" || a.startsWith("pcm_")))
            s.audio_codec = "aac";
    }

    s.cq = std::clamp(s.cq, 1, 51);

    applyAv1Restriction();
    updateTileAllocation();
    applyRateRange();
}

// ==============================================================
// 選択肢
// ==============================================================
QVariantList EncodeController::containerOptions() const
{
    return { opt("MP4 (.mp4)", "mp4"), opt("MKV (.mkv)", "mkv"), opt("MOV (.mov)", "mov") };
}

QVariantList EncodeController::codecOptions() const
{
    QVariantList l;
    for (const char* c : {"h264_nvenc", "hevc_nvenc", "av1_nvenc"})
        if (codecAllowed(c)) l << opt(c, QString(c));
    return l;
}

QVariantList EncodeController::framerateOptions() const
{
    QList<double> v = {1, 2, 5, 10, 15, 20, 24, 30, 60, 120, 240, 300};
    if (VideoInfo.fps > 0 && !v.contains(VideoInfo.fps)) v << VideoInfo.fps;
    if (es().save_fps > 0 && !v.contains(es().save_fps)) v << es().save_fps;
    std::sort(v.begin(), v.end());
    QVariantList l;
    for (double f : v) l << opt(QString::number(f), f);
    return l;
}

QVariantList EncodeController::splitOptions() const
{
    return { opt("Disable", "0"), opt("Enable", "1") };
}

QVariantList EncodeController::tileOptions() const
{
    QVariantList l;
    const struct { const char* t; int v; } items[] = {
        {"1 Tile (通常)", 1}, {"2 Tiles (2 × 1)", 2}, {"4 Tiles (2 × 2)", 4}, {"8 Tiles (4 × 2)", 8} };
    for (const auto& i : items)
        if (tileAllowed(i.v)) l << opt(QString::fromUtf8(i.t), i.v);
    return l;
}

QVariantList EncodeController::presetOptions() const
{
    QVariantList l;
    for (int i = 1; i <= 7; i++) {
        QString t = QString("p%1").arg(i);
        if (i == 1) t += " (最速・低品質)";
        if (i == 7) t += " (最高品質・低速)";
        l << opt(t, QString("p%1").arg(i));
    }
    return l;
}

QVariantList EncodeController::tuneOptions() const
{
    QVariantList l;
    const struct { const char* t; const char* v; } items[] = {
        {"default (汎用)", "default"}, {"hq (高品質)", "hq"}, {"ll (低遅延)", "ll"},
        {"ull (超低遅延)", "ull"}, {"lossless (可逆圧縮)", "lossless"} };
    for (const auto& i : items)
        if (tuneAllowed(i.v)) l << opt(QString::fromUtf8(i.t), QString(i.v));
    return l;
}

QVariantList EncodeController::gopOptions() const
{
    QVariantList l;
    for (int g : {1, 15, 30, 60, 120, 250, 300}) l << opt(QString::number(g), g);
    return l;
}

QVariantList EncodeController::bframeOptions() const
{
    QVariantList l;
    for (int b = 0; b <= 7; b++)
        if (bframeAllowed(b)) l << opt(QString::number(b), b);
    return l;
}

QVariantList EncodeController::rcOptions() const
{
    return { opt("CBR", "cbr"), opt("VBR", "vbr"), opt("CQP", "cqp"), opt("CQ (品質指定VBR)", "cq") };
}

QVariantList EncodeController::passOptions() const
{
    return { opt("1 pass", "1pass"), opt("2 pass quarter", "2pass-quarter-res"), opt("2 pass full", "2pass-full-res") };
}

QVariantList EncodeController::basisOptions() const
{
    return { opt("1 ストリームあたり", "stream"), opt("合計（ファイル全体）", "total") };
}

// ★ コンテナに入る音声コーデックだけ
//   MP4 : PCM 不可（ipcm になり再生環境を選ぶ）／ MOV : FLAC 不可
QVariantList EncodeController::audioCodecOptions() const
{
    const QString c = es().container;
    QVariantList l;
    l << opt("AAC (非可逆)", "aac");
    if (c != "mov") l << opt("FLAC (可逆圧縮)", "flac");
    if (c != "mp4") {
        l << opt("PCM 16bit (非圧縮)", "pcm_s16le");
        l << opt("PCM 24bit (非圧縮)", "pcm_s24le");
        l << opt("PCM 32bit float (非圧縮)", "pcm_f32le");
    }
    return l;
}

// ==============================================================
// 値の変更
// ==============================================================
void EncodeController::setValue(const QString& key, const QVariant& value)
{
    if (encoding_) return;
    EncodeSettings& s = es();

    if (key == "container") {
        s.container = value.toString();
        applyContainerToPath();
    } else if (key == "codec") {
        s.codec = value.toString();
    } else if (key == "framerate") {
        s.save_fps = value.toDouble();
    } else if (key == "split") {
        s.split_encode_mode = value.toString();
    } else if (key == "tile") {
        // ★ 合計基準のときは「合計値は据え置き＝1 本あたりが変わる」
        s.encode_tile = value.toInt();
    } else if (key == "preset") {
        s.preset = value.toString();
    } else if (key == "tune") {
        s.tune = value.toString();
    } else if (key == "gop") {
        s.gop_size = value.toInt();
    } else if (key == "bframes") {
        s.b_frames = value.toInt();
    } else if (key == "rc") {
        s.rc_mode = value.toString();
    } else if (key == "pass") {
        s.pass_mode = value.toString();
    } else if (key == "basis") {
        // ★ 単位を変えても実際のビットレートは変えない。表示値だけ換算する
        const QString next = (value.toString() == "total") ? "total" : "stream";
        if (next != s.bitrate_basis) {
            const int tiles = std::max(1, s.encode_tile);
            if (next == "total") { target_ *= tiles; max_ *= tiles; }
            else { target_ = std::max(1, target_ / tiles); max_ = std::max(1, max_ / tiles); }
            s.bitrate_basis = next;
        }
    } else if (key == "audioCodec") {
        s.audio_codec = value.toString();
    } else {
        qWarning() << "[EncodeController] unknown key" << key;
        return;
    }
    normalize();
    emit changed();
}

void EncodeController::setSeparateFiles(bool on)
{
    if (encoding_ || es().separate_files == on) return;
    es().separate_files = on;
    emit changed();
}

void EncodeController::setAudioEnabled(bool on)
{
    if (encoding_ || es().audio_enable == on) return;
    es().audio_enable = on;
    emit changed();
}

void EncodeController::setTargetRate(int v)
{
    if (encoding_) return;
    target_ = std::clamp(v, rateMin(), targetHi());
    emit changed();
}

void EncodeController::setMaxRate(int v)
{
    if (encoding_) return;
    max_ = std::clamp(v, rateMin(), rateHi());
    // VBR は 目標 <= 最大
    if (es().rc_mode == "vbr" && target_ > max_) target_ = max_;
    emit changed();
}

void EncodeController::setQuality(int v)
{
    if (encoding_) return;
    es().cq = std::clamp(v, 1, 51);
    emit changed();
}

void EncodeController::setAudioBitrate(int kbps)
{
    if (encoding_) return;
    // 刻みは 16 kbps
    kbps = std::clamp((kbps + 8) / 16 * 16, 64, 512);
    es().audio_bitrate = kbps * 1000;
    emit changed();
}

void EncodeController::setAudioGain(int db)
{
    if (encoding_) return;
    es().audio_gain_db = std::clamp(db, -30, 30);
    emit changed();
}

// ★ 1 本あたりが常に 1〜1000 Mbps に収まるよう範囲を決め直す（旧 applyBitrateRange）
void EncodeController::applyRateRange()
{
    max_    = std::clamp(max_, rateMin(), rateHi());
    target_ = std::clamp(target_, rateMin(), targetHi());
}

// ==============================================================
// ビットレート / 品質の表示
// ==============================================================
bool EncodeController::showTarget() const  { const QString r = es().rc_mode; return r == "cbr" || r == "vbr"; }
bool EncodeController::showMax() const     { const QString r = es().rc_mode; return r == "vbr" || r == "cq"; }
bool EncodeController::showQuality() const { const QString r = es().rc_mode; return r == "cq"  || r == "cqp"; }

QString EncodeController::qualityLabel() const { return es().rc_mode == "cqp" ? "量子化 (QP)" : "目標品質 (CQ)"; }
QString EncodeController::qualityFlag() const  { return es().rc_mode == "cqp" ? "-qp" : "-cq"; }

QString EncodeController::rateUnit() const
{
    const int tiles = std::max(1, es().encode_tile);
    return (es().bitrate_basis == "total" || tiles == 1) ? "Mbps" : "Mbps/本";
}

// 換算は常に 1 本あたり（NVENC が受け取るのがその値）
static QString rateLine(const QString& name, int shown, int unitDiv, int tiles)
{
    const int ps = std::max(1, shown / unitDiv);
    return (tiles <= 1) ? QString("%1 %2 Mbps").arg(name).arg(ps)
                        : QString("%1 %2 Mbps/本 × %3 タイル = 合計 %4 Mbps")
                              .arg(name).arg(ps).arg(tiles).arg((qint64)ps * tiles);
}

QString EncodeController::rateInfoTarget() const
{
    return showTarget() ? rateLine("目標", target_, unit(), std::max(1, es().encode_tile)) : QString();
}

QString EncodeController::rateInfoMax() const
{
    return showMax() ? rateLine("最大", max_, unit(), std::max(1, es().encode_tile)) : QString();
}

// ★ 式は __global__.h（save_encode と共有。表示と実体をずらさない）
QString EncodeController::vramInfo() const
{
    const EncodeSettings& s = es();
    const int tiles = std::max(1, s.encode_tile);
    const int bf   = std::clamp(std::min(s.b_frames, nvenc_max_bframes(s.codec)), 0, 7);
    const int ring = encode_ring_size(bf);
    const int tw   = canvasW() / std::max(1, s.width_tile);
    const int th   = canvasH() / std::max(1, s.height_tile);
    const int mb   = encode_pool_mb(tw, th, ring, tiles);
    return QString("想定使用 VRAM %1 MB   (%2 x %3 x %4 面 x %5 タイル)")
        .arg(mb).arg(tw).arg(th).arg(ring + 4).arg(tiles);
}

QString EncodeController::summary() const
{
    const EncodeSettings& s = es();
    const int tw = canvasW() / std::max(1, s.width_tile);
    const int th = canvasH() / std::max(1, s.height_tile);
    return QString("%1×%2 · %3fps  →  %4 · %5 タイル · %6×%7/本")
        .arg(canvasW()).arg(canvasH()).arg(VideoInfo.fps)
        .arg(s.codec).arg(s.encode_tile).arg(tw).arg(th);
}

// ==============================================================
// 音声
// ==============================================================
bool EncodeController::audioAvailable() const { return VideoInfo.audio; }

bool EncodeController::audioLossless() const
{
    const QString a = es().audio_codec;
    return a.startsWith("pcm_") || a == "flac";
}

// ==============================================================
// ファイル
// ==============================================================
void EncodeController::browsePath()
{
    if (encoding_) return;
    const QString ext = es().container;
    const QString filter = QString("%1 Files (*.%2);;All Files (*.*)").arg(ext.toUpper(), ext);
    const QFileInfo fi(es().encode_path);
    const QString fileName = QFileDialog::getSaveFileName(nullptr, tr("名前を付けて保存"),
                                                          fi.absoluteFilePath(), filter);
    if (fileName.isEmpty()) return;
    es().encode_path = fileName;
    // ★ 別の拡張子で保存されたらコンテナをそちらへ合わせる（パスのほうが正）
    syncContainerFromPath();
    normalize();
    emit changed();
}

void EncodeController::applyContainerToPath()
{
    const QFileInfo fi(es().encode_path);
    es().encode_path = fi.absolutePath() + "/" + fi.completeBaseName() + "." + es().container;
}

// 知らない拡張子のときはコンテナを正として付け直す
void EncodeController::syncContainerFromPath()
{
    const QString ext = QFileInfo(es().encode_path).suffix().toLower();
    if (ext == "mp4" || ext == "mkv" || ext == "mov") es().container = ext;
    else applyContainerToPath();
}

// ファイルの重複チェック（旧 file_check と同じ）
QString EncodeController::fileCheck(const QString& filePath)
{
    const QFileInfo fileInfo(filePath);
    const QString dir = fileInfo.absolutePath();
    const QString baseName = fileInfo.completeBaseName();
    const QString ext = fileInfo.suffix();

    QString baseNameOnly = baseName;
    int counter = 1;
    const QRegularExpression re(R"(^(.*)\((\d+)\)$)");
    const QRegularExpressionMatch match = re.match(baseName);
    if (match.hasMatch()) {
        baseNameOnly = match.captured(1);
        counter = match.captured(2).toInt();
    }

    QString newPath = filePath;
    const QDir qDir(dir);
    while (QFile::exists(newPath))
        newPath = qDir.filePath(QString("%1(%2).%3").arg(baseNameOnly).arg(counter++).arg(ext));
    return newPath;
}

// ==============================================================
// マルチ GPU（旧 setGPUTable / updateGpuTileAllocation / buildTileGpuMap）
// ==============================================================
void EncodeController::applyAv1Restriction()
{
    const bool av1 = (es().codec == "av1_nvenc") && primaryAv1();
    for (auto& g : g_GPUInfo) {
        if (av1) {
            // セカンダリで AV1 非対応のものは使わない（重みは退避しておく）
            if (!g.openglEnable && !av1Capable(g)) {
                if (g.backup_tile_weight < 0) g.backup_tile_weight = g.tile_weight;
                g.av1_forced_disabled = true;
                g.tile_weight = 0;
            }
        } else if (g.backup_tile_weight >= 0) {
            g.tile_weight = g.backup_tile_weight;
            g.backup_tile_weight = -1;
            g.av1_forced_disabled = false;
        }
    }
}

void EncodeController::updateTileAllocation()
{
    // 並びは Primary（OpenGL）を先頭に
    std::vector<int> order;
    for (int i = 0; i < (int)g_GPUInfo.size(); i++) if (g_GPUInfo[i].openglEnable)  order.push_back(i);
    for (int i = 0; i < (int)g_GPUInfo.size(); i++) if (!g_GPUInfo[i].openglEnable) order.push_back(i);

    const int totalTiles = std::max(1, es().encode_tile);
    int totalWeight = 0;
    for (int i : order) if (g_GPUInfo[i].tile_weight > 0) totalWeight += g_GPUInfo[i].tile_weight;
    if (totalWeight <= 0) totalWeight = 1;

    gpu_tiles_.clear();
    int used = 0;
    for (int i : order) {
        const int w = g_GPUInfo[i].tile_weight;
        const int t = (w > 0) ? (w * totalTiles) / totalWeight : 0;
        used += t;
        gpu_tiles_.push_back({ g_GPUInfo[i].deviceID, t });
    }
    // 余りは weight>0 の行へ上から 1 枚ずつ
    int remain = totalTiles - used;
    for (size_t r = 0; r < gpu_tiles_.size() && remain > 0; r++) {
        if (g_GPUInfo[order[r]].tile_weight <= 0) continue;
        gpu_tiles_[r].second++;
        remain--;
    }
}

// 表示用の GPU 名（CUDA 必須アプリなので先頭の "NVIDIA " は省く）
static QString displayGpuName(QString name)
{
    name = name.trimmed();
    if (name.startsWith(QLatin1String("NVIDIA "), Qt::CaseInsensitive))
        name.remove(0, 7);
    return name;
}

QVariantList EncodeController::gpuStats() const
{
    QVariantList list;
    for (const auto& [id, tiles] : gpu_tiles_) {
        (void)tiles;
        const GPUInfo* g = nullptr;
        for (const auto& x : g_GPUInfo) if (x.deviceID == id) { g = &x; break; }
        if (!g) continue;   // gpus() と同じ条件で飛ばす → 並びが一致する
        QVariantMap m;
        m["encUsage"] = g->Encoder_Usage;
        m["gpuUsage"] = g->GPU_Usage;
        m["memUsed"]  = g->Memory_Usage;
        m["memMax"]   = g->Max_Memory_Usage;
        list << m;
    }
    return list;
}

QVariantList EncodeController::gpus() const
{
    QVariantList list;
    const int totalTiles = std::max(1, es().encode_tile);
    for (const auto& [id, tiles] : gpu_tiles_) {
        const GPUInfo* g = nullptr;
        for (const auto& x : g_GPUInfo) if (x.deviceID == id) { g = &x; break; }
        if (!g) continue;

        QVariantMap m;
        m["id"]       = id;
        m["name"]     = displayGpuName(g->deviceName);
        m["memory"]   = QString("%1 / %2").arg(g->Memory_Usage).arg(g->Max_Memory_Usage);
        m["primary"]  = g->openglEnable;
        m["weight"]   = g->tile_weight;
        m["minWeight"] = g->openglEnable ? 1 : 0;   // Primary は必ず使う
        m["locked"]   = g->av1_forced_disabled;
        m["tiles"]    = tiles;

        QString status;
        bool used = true;
        if (g->av1_forced_disabled)      { status = "Not Used (AV1 非対応)"; used = false; }
        else if (g->tile_weight <= 0)    { status = "Not Used"; used = false; }
        else if (tiles <= 0)             { status = "0 (0%)"; used = false; }
        else status = QString("%1 (%2%)").arg(tiles).arg(int(double(tiles) / totalTiles * 100.0 + 0.5));
        m["status"] = status;
        m["used"]   = used;
        list << m;
    }
    return list;
}

void EncodeController::setGpuWeight(int deviceId, int weight)
{
    if (encoding_) return;
    for (auto& g : g_GPUInfo) {
        if (g.deviceID != deviceId) continue;
        if (g.av1_forced_disabled) return;
        g.tile_weight = std::clamp(weight, g.openglEnable ? 1 : 0, 10);
        break;
    }
    updateTileAllocation();
    emit changed();
}

std::vector<int> EncodeController::buildTileGpuMap() const
{
    std::vector<int> map;
    const int totalTiles = std::max(1, es().encode_tile);
    for (const auto& [id, tiles] : gpu_tiles_)
        for (int t = 0; t < tiles; t++) map.push_back(id);
    while ((int)map.size() < totalTiles) map.push_back(g_openglDeviceID);   // 念のため不足補正
    if ((int)map.size() > totalTiles) map.resize(totalTiles);
    return map;
}

// ==============================================================
// 開始・キャンセル・完了
// ==============================================================
void EncodeController::prepare(int progressMax)
{
    progress_max_ = std::max(1, progressMax);
    progress_ = 0;
    status_.clear();
    finished_ok_ = false;
    encoding_ = false;
    // 保存フレームレートの初期値は素材の fps
    es().save_fps = VideoInfo.fps;
    statsTimer_.start();
    emit statsChanged();
    normalize();
    emit changed();
    emit stateChanged();
    emit progressChanged();
}

void EncodeController::setProgress(int value)
{
    if (value == progress_) return;
    progress_ = std::clamp(value, 0, progress_max_);

    // ★ エンコード速度：0.5 秒ごとに「その区間で進んだフレーム数 / 経過時間」
    if (encoding_) {
        if (!fpsTimer_.isValid()) { fpsTimer_.start(); fpsBase_ = progress_; }
        const qint64 ms = fpsTimer_.elapsed();
        if (ms >= 500) {
            encode_fps_ = (progress_ - fpsBase_) * 1000.0 / ms;
            fpsBase_ = progress_;
            fpsTimer_.restart();
        }
    }
    emit progressChanged();
}

void EncodeController::start()
{
    if (encoding_) return;
    EncodeSettings& s = es();

    // ★ 保存は常に 1 ストリームあたり。合計基準ならタイル数で割り戻す
    const int u = unit();
    qint64 tgt = (qint64)target_ / u;
    qint64 mx  = (s.rc_mode == "cbr") ? tgt : ((qint64)max_ / u);
    tgt = std::max<qint64>(1, tgt) * 1000 * 1000;
    mx  = std::max<qint64>(1, mx)  * 1000 * 1000;
    // ★ target と max は別々に通す
    const qint64 cap = codec_max_bitrate(s.codec);
    s.target_bit_rate = (int)std::min(tgt, cap);
    s.max_bit_rate    = (int)std::min(mx,  cap);

    // 保存先フォルダがあるか
    if (!QDir(QFileInfo(s.encode_path).absolutePath()).exists()) {
        status_ = "保存先のフォルダがありません";
        finished_ok_ = false;
        emit stateChanged();
        return;
    }
    // 入力ファイルへの上書きは避ける
    if (QString::fromStdString(VideoInfo.Path) == s.encode_path)
        s.encode_path = fileCheck(s.encode_path);

    s.tile_gpu_map = buildTileGpuMap();

    encoding_ = true;
    canceled_ = false;
    finished_ok_ = false;
    progress_ = 0;
    encode_fps_ = 0.0;
    fpsTimer_.invalidate();
    totalTimer_.start();
    status_ = "エンコード中";
    emit changed();
    emit stateChanged();
    emit progressChanged();
    emit encodeStartRequested();
}

void EncodeController::cancel()
{
    if (!encoding_) return;
    canceled_ = true;
    emit encodeStopRequested();
    encoding_ = false;
    status_ = "キャンセルしました";
    emit changed();
    emit stateChanged();
}

void EncodeController::encodeEnded(const QString& elapsed)
{
    // ★ キャンセルしても途中までのファイルは閉じて書き終える（旧実装と同じ）
    encoding_ = false;
    finished_ok_ = !canceled_;
    if (!canceled_) progress_ = progress_max_;
    // 平均速度 = 書き出したフレーム数 / 開始からの時間
    const double sec = totalTimer_.isValid() ? totalTimer_.nsecsElapsed() / 1e9 : 0.0;
    const QString avg = sec > 0.0 ? QString(" · 平均 %1 fps").arg(progress_ / sec, 0, 'f', 2) : QString();
    encode_fps_ = 0.0;
    fpsTimer_.invalidate();
    status_ = canceled_ ? QString("キャンセルしました · 途中まで書き出し済み（%1%2）").arg(elapsed, avg)
                        : QString("完了 · 処理時間 %1%2").arg(elapsed, avg);
    canceled_ = false;
    updateTileAllocation();
    emit changed();
    emit stateChanged();
    emit progressChanged();
}

void EncodeController::encodeFailed(const QString& message)
{
    encoding_ = false;
    finished_ok_ = false;
    progress_ = 0;
    status_ = QString("開始できませんでした: %1").arg(message);
    emit changed();
    emit stateChanged();
    emit progressChanged();
}

bool EncodeController::close()
{
    if (encoding_) return false;
    statsTimer_.stop();
    emit closed();
    return true;
}
