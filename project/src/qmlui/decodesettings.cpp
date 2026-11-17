#include "src/qmlui/decodesettings.h"
#include "src/main/__global__.h"
#include "src/videoprocess/decode_gpu_map.h"

#include <QVariantMap>
#include <set>

DecodeSettings::DecodeSettings(QObject* parent)
    : QObject(parent)
{
}

// GPU 名から先頭の "NVIDIA " を省く（他の画面と同じ）
static QString shortGpuName(QString n)
{
    if (n.startsWith("NVIDIA ")) n.remove(0, 7);
    return n;
}

// 見込みに使うコーデック名（decode_gpu_map.h の gpuCanDecode と同じ名前に）
static std::string codecKey(const QString& c)
{
    if (c.contains("av1"))  return "av1";
    if (c.contains("hevc")) return "hevc";
    if (c.contains("h264")) return "h264";
    return std::string();
}

// g_GPUInfo に編集中の重みを当てたもの（割り振りの見込み用）
static std::vector<GPUInfo> editedGpus(const std::map<int, int>& weights)
{
    std::vector<GPUInfo> tmp = g_GPUInfo;
    for (GPUInfo& g : tmp) {
        auto it = weights.find(g.deviceID);
        if (it != weights.end()) g.decode_weight = it->second;
    }
    return tmp;
}

bool DecodeSettings::multiGpu() const
{
    // 開いているファイルがあればその割り振りで、無ければ「重みが 1 以上でデコードできる GPU」の数で判断する
    const std::vector<GPUInfo> tmp = editedGpus(weights_);
    const std::string key = codecKey(fileCodec_);
    const char* codec = key.empty() ? nullptr : key.c_str();
    std::set<int> used;
    if (fileTiles_ > 0) {
        for (int id : assignDecodeTiles(fileTiles_, codec, tmp)) used.insert(id);
    } else {
        for (const GPUInfo& g : tmp)
            if (effectiveDecodeWeight(g) > 0 && gpuCanDecode(g, codec)) used.insert(g.deviceID);
    }
    return used.size() > 1;
}

QVariantList DecodeSettings::gpus() const
{
    const std::vector<GPUInfo> tmp = editedGpus(weights_);
    const std::string key = codecKey(fileCodec_);
    const char* codec = key.empty() ? nullptr : key.c_str();
    std::map<int, int> tiles;
    if (fileTiles_ > 0)
        for (int id : assignDecodeTiles(fileTiles_, codec, tmp)) tiles[id]++;

    std::vector<const GPUInfo*> order;   // 表示用 GPU を先頭に、残りは番号順
    for (const GPUInfo& g : tmp) order.push_back(&g);
    std::stable_sort(order.begin(), order.end(), [](const GPUInfo* a, const GPUInfo* b) {
        const bool pa = a->deviceID == g_openglDeviceID, pb = b->deviceID == g_openglDeviceID;
        if (pa != pb) return pa;
        return a->deviceID < b->deviceID;
    });

    QVariantList list;
    for (const GPUInfo* g : order) {
        QVariantMap m;
        m["id"]        = g->deviceID;
        m["name"]      = shortGpuName(g->deviceName);
        m["sm"]        = QString("sm_%1%2").arg(g->CC_major).arg(g->CC_minor);
        m["weight"]    = effectiveDecodeWeight(*g);
        m["canDecode"] = gpuCanDecode(*g, codec);
        m["primary"]   = (g->deviceID == g_openglDeviceID);
        m["dec"]       = g->Decoder_Usage;
        m["tiles"]     = fileTiles_ > 0 ? tiles[g->deviceID] : -1;
        list << m;
    }
    return list;
}

QVariantList DecodeSettings::xferCodecOptions() const
{
    auto opt = [](const char* t, const char* v) { return QVariantMap{ {"text", QString::fromUtf8(t)}, {"value", QString::fromUtf8(v)} }; };
    return { opt("なし", "none"), opt("ANS（推奨）", "ans"), opt("Cascaded", "cascaded"), opt("Bitcomp", "bitcomp") };
}

void DecodeSettings::setXferCodec(const QString& v)
{
    if (v == xferCodec_ || (v != "none" && v != "cascaded" && v != "bitcomp" && v != "ans")) return;
    xferCodec_ = v;
    emit changed();
}

void DecodeSettings::setXferDelta(bool on)
{
    if (on == xferDelta_) return;
    xferDelta_ = on;
    emit changed();
}

bool DecodeSettings::dirty() const
{
    if (xferCodec_ != g_AppSettings.decode_xfer_codec || xferDelta_ != g_AppSettings.decode_xfer_delta) return true;
    for (const GPUInfo& g : g_GPUInfo) {
        auto it = weights_.find(g.deviceID);
        if (it != weights_.end() && it->second != effectiveDecodeWeight(g)) return true;
    }
    return false;
}

void DecodeSettings::open(int tiles, const QString& codec)
{
    weights_.clear();
    for (const GPUInfo& g : g_GPUInfo) weights_[g.deviceID] = effectiveDecodeWeight(g);
    fileTiles_ = tiles;
    fileCodec_ = codec;
    xferCodec_ = g_AppSettings.decode_xfer_codec;
    xferDelta_ = g_AppSettings.decode_xfer_delta;
    emit changed();
    emit opened();
}

void DecodeSettings::setWeight(int deviceId, int weight)
{
    weight = qBound(0, weight, 10);
    auto it = weights_.find(deviceId);
    if (it == weights_.end() || it->second == weight) return;
    it->second = weight;
    emit changed();
}

void DecodeSettings::apply(bool reopen)
{
    const bool wasDirty = dirty();
    g_AppSettings.decode_xfer_codec = xferCodec_;
    g_AppSettings.decode_xfer_delta = xferDelta_;
    for (GPUInfo& g : g_GPUInfo) {
        auto it = weights_.find(g.deviceID);
        if (it != weights_.end()) g.decode_weight = it->second;
    }
    emit closed();
    if (reopen && wasDirty) emit reopenRequested();
}

void DecodeSettings::cancel()
{
    emit closed();
}
