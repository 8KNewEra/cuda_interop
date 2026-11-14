#include "src/qmlui/screenshotsettings.h"
#include "src/main/__global__.h"

#include <QDir>
#include <QFileDialog>
#include <QStandardPaths>
#include <QVariantMap>

// 並び順と設定値の対応（Widgets 版 save_screenshot.cpp と同じ表）
namespace {
struct Opt { const char* label; const char* value; };

const Opt kFormats[] = {
    {"JPG (.jpg)",   "jpg"},
    {"PNG (.png)",   "png"},
    {"TIFF (.tiff)", "tiff"},
    {"BMP (.bmp)",   "bmp"},
};
const Opt kTiffCompressions[] = {
    {"可逆 (Deflate)", "deflate"},
    {"非圧縮",         "none"},
};
// 4:4:4 = 色差を間引かない（文字・細線・色の境界がにじまない、ファイルは大きい）
// 4:2:0 = 縦横とも半分（一般的な写真・動画と同じ、最も小さい）
const Opt kJpegSubsamplings[] = {
    {"4:4:4 (高画質)", "444"},
    {"4:2:2",          "422"},
    {"4:2:0 (標準)",   "420"},
};
// PNG / TIFF のビット深度。"16" は 10bit の値を 16bit のファイルに入れる（表記は 10bit）
//   10bit を選べるのは 10bit 以上の動画のときだけ（depthOptions）
const Opt kDepths[] = {
    {"10bit", "16"},
    {"8bit",  "8"},
};
// 10bit 以上の動画を 8bit で保存するときのディザ（縞を抑える / 値そのまま）
const Opt kDithers[] = {
    {"あり（なめらか）",     "on"},
    {"なし（値そのまま）",   "off"},
};
const Opt kModes[] = {
    {"fast",   "fast"},
    {"normal", "normal"},
    {"high",   "high"},
};

template <size_t N>
QVariantList toOptions(const Opt (&table)[N])
{
    QVariantList list;
    for (const Opt& o : table)
        list << QVariantMap{ {"text", QString::fromUtf8(o.label)}, {"value", QString::fromUtf8(o.value)} };
    return list;
}

// 表に無い値なら fallback（AppSettings の既定値）、それも無ければ先頭
template <size_t N>
QString normalized(const Opt (&table)[N], const QString& value, const QString& fallback)
{
    for (const Opt& o : table) if (value == o.value) return value;
    for (const Opt& o : table) if (fallback == o.value) return fallback;
    return QString::fromUtf8(table[0].value);
}

template <size_t N>
bool contains(const Opt (&table)[N], const QString& value)
{
    for (const Opt& o : table) if (value == o.value) return true;
    return false;
}

// 保存先フォルダが無効なら 画像フォルダ → ホーム の順に補う
QString validFolder(const QString& folder)
{
    if (!folder.isEmpty() && QDir(folder).exists()) return folder;
    const QString pictures = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    if (!pictures.isEmpty() && QDir(pictures).exists()) return pictures;
    return QDir::homePath();
}
} // namespace

ScreenshotSettings::ScreenshotSettings(QObject* parent)
    : QObject(parent)
{
    // 起動時（ini 読み込み直後）に一度だけ正規化した値を g_AppSettings へ戻す
    // これで保存処理側は常に有効な値だけを見ればよい
    loadFromGlobal();
    commitToGlobal();
}

QVariantList ScreenshotSettings::formatOptions() const          { return toOptions(kFormats); }
QVariantList ScreenshotSettings::tiffCompressionOptions() const { return toOptions(kTiffCompressions); }
QVariantList ScreenshotSettings::jpegSubsamplingOptions() const { return toOptions(kJpegSubsamplings); }
QVariantList ScreenshotSettings::modeOptions() const            { return toOptions(kModes); }
QVariantList ScreenshotSettings::ditherOptions() const          { return toOptions(kDithers); }
// 8bit の動画のときは 8bit だけ（希望の値 depth_ は残すので、10bit の動画では元に戻る）
QVariantList ScreenshotSettings::depthOptions() const
{
    if (srcHighBit_) return toOptions(kDepths);
    return { QVariantMap{ {"text", "8bit"}, {"value", "8"} } };
}

void ScreenshotSettings::loadFromGlobal()
{
    const AppSettings def;   // 既定値は __global__.h の 1 か所だけで決める
    folder_          = validFolder(g_AppSettings.saveScreenshot_folder);
    format_          = normalized(kFormats,          g_AppSettings.screenshot_format,           def.screenshot_format);
    tiffCompression_ = normalized(kTiffCompressions, g_AppSettings.screenshot_tiff_compression, def.screenshot_tiff_compression);
    jpegQuality_     = qBound(0, g_AppSettings.screenshot_jpeg_quality, 100);
    jpegSubsampling_ = normalized(kJpegSubsamplings, g_AppSettings.screenshot_jpeg_subsampling, def.screenshot_jpeg_subsampling);
    mode_            = normalized(kModes,            g_AppSettings.screenshot_mode,             def.screenshot_mode);
    depth_           = normalized(kDepths,           g_AppSettings.screenshot_depth,            def.screenshot_depth);
    dither_          = normalized(kDithers,          g_AppSettings.screenshot_dither,           def.screenshot_dither);
}

void ScreenshotSettings::commitToGlobal()
{
    g_AppSettings.saveScreenshot_folder       = folder_;
    g_AppSettings.screenshot_format           = format_;
    g_AppSettings.screenshot_tiff_compression = tiffCompression_;
    g_AppSettings.screenshot_jpeg_quality     = jpegQuality_;
    g_AppSettings.screenshot_jpeg_subsampling = jpegSubsampling_;
    g_AppSettings.screenshot_mode             = mode_;
    g_AppSettings.screenshot_depth            = depth_;
    g_AppSettings.screenshot_dither           = dither_;
}

void ScreenshotSettings::open()
{
    loadFromGlobal();
    emit changed();
    emit opened();
}

void ScreenshotSettings::setValue(const QString& key, const QVariant& value)
{
    const QString v = value.toString();
    if      (key == "format"          && contains(kFormats, v))          format_ = v;
    else if (key == "tiffCompression" && contains(kTiffCompressions, v)) tiffCompression_ = v;
    else if (key == "jpegSubsampling" && contains(kJpegSubsamplings, v)) jpegSubsampling_ = v;
    else if (key == "mode"            && contains(kModes, v))            mode_ = v;
    else if (key == "depth"           && contains(kDepths, v))           depth_ = v;
    else if (key == "dither"          && contains(kDithers, v))          dither_ = v;
    else return;
    emit changed();
}

void ScreenshotSettings::setJpegQuality(int value)
{
    value = qBound(0, value, 100);
    if (value == jpegQuality_) return;
    jpegQuality_ = value;
    emit changed();
}

void ScreenshotSettings::pickFolder()
{
    // OK を押すまでは編集中の値だけ変える
    const QString dir = QFileDialog::getExistingDirectory(
        nullptr, tr("フォルダーを選択"), validFolder(folder_), QFileDialog::ShowDirsOnly);
    if (dir.isEmpty()) return;
    folder_ = dir;
    emit changed();
}

void ScreenshotSettings::apply()
{
    commitToGlobal();
    emit closed();
}

void ScreenshotSettings::cancel()
{
    emit closed();
}
