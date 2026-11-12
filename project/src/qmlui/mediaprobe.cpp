#include "src/qmlui/mediaprobe.h"

#include <QClipboard>
#include <QDir>
#include <QHash>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QTextStream>
#include <QVariantMap>
#include <QtConcurrent/QtConcurrent>
#include <cmath>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/pixdesc.h>
}

namespace {

using Rows = QVariantList;

void add(Rows& rows, const QString& k, const QString& v)
{
    if (!v.isEmpty()) rows << QVariantMap{ {"k", k}, {"v", v} };
}

QString fmtSize(int64_t bytes)
{
    if (bytes <= 0) return {};
    const double b = double(bytes);
    QString s;
    if (b >= 1024.0 * 1024 * 1024) s = QString::number(b / (1024.0 * 1024 * 1024), 'f', 2) + " GiB";
    else if (b >= 1024.0 * 1024)   s = QString::number(b / (1024.0 * 1024), 'f', 1) + " MiB";
    else                           s = QString::number(b / 1024.0, 'f', 1) + " KiB";
    return s + QString("（%L1 バイト）").arg(bytes);
}

QString fmtDuration(double sec)
{
    if (!(sec > 0)) return {};
    const int64_t ms = int64_t(std::llround(sec * 1000.0));
    const int h = int(ms / 3600000), m = int(ms / 60000 % 60), s = int(ms / 1000 % 60), f = int(ms % 1000);
    return QString("%1:%2:%3.%4").arg(h, 2, 10, QChar('0')).arg(m, 2, 10, QChar('0'))
                                 .arg(s, 2, 10, QChar('0')).arg(f, 3, 10, QChar('0'));
}

QString fmtBitrate(int64_t bps)
{
    if (bps <= 0) return {};
    if (bps >= 1000000) return QString::number(bps / 1e6, 'f', bps >= 100000000 ? 0 : 1) + " Mbps";
    return QString::number(bps / 1e3, 'f', 0) + " kbps";
}

QString fmtRational(AVRational r, int prec = 3)
{
    if (r.num <= 0 || r.den <= 0) return {};
    return QString::number(av_q2d(r), 'f', prec);
}

double streamDuration(const AVFormatContext* fc, const AVStream* st)
{
    if (st->duration != AV_NOPTS_VALUE && st->duration > 0) return st->duration * av_q2d(st->time_base);
    if (fc->duration != AV_NOPTS_VALUE && fc->duration > 0) return fc->duration / double(AV_TIME_BASE);
    return 0.0;
}

// よく出るタグは日本語の項目名にする（それ以外はタグ名のまま）
QString tagLabel(const QString& key)
{
    static const QHash<QString, QString> k = {
        {"title", "タイトル"}, {"comment", "コメント"}, {"description", "説明"}, {"artist", "作成者"},
        {"copyright", "著作権"}, {"date", "日付"}, {"timecode", "タイムコード"}, {"creation_time", "作成日時"},
        {"encoder", "エンコーダ"}, {"handler_name", "ハンドラ名"}, {"rotate", "回転（タグ）"},
    };
    return k.value(key.toLower(), key);
}

void addTags(Rows& rows, const AVDictionary* dict, const QStringList& skip = {})
{
    const AVDictionaryEntry* e = nullptr;
    while ((e = av_dict_iterate(dict, e))) {
        const QString key = QString::fromUtf8(e->key);
        const QString val = QString::fromUtf8(e->value);
        if (skip.contains(key, Qt::CaseInsensitive)) continue;
        if (key == "vendor_id" && val == "[0][0][0][0]") continue;   // 中身のない既定値
        add(rows, tagLabel(key), val);
    }
}

QString tag(const AVDictionary* dict, const char* key)
{
    const AVDictionaryEntry* e = av_dict_get(dict, key, nullptr, 0);
    return e ? QString::fromUtf8(e->value) : QString();
}

QString profileOf(const AVCodecParameters* p)
{
    const char* prof = avcodec_profile_name(p->codec_id, p->profile);
    QString s = prof ? QString::fromUtf8(prof) : QString();
    if (p->level > 0) {
        // H.264 / HEVC / AV1 は表記の流儀が違う
        QString lv;
        if (p->codec_id == AV_CODEC_ID_H264)      lv = QString::number(p->level / 10.0, 'f', 1);
        else if (p->codec_id == AV_CODEC_ID_HEVC) lv = QString::number(p->level / 30.0, 'f', 1);
        else if (p->codec_id == AV_CODEC_ID_AV1)  lv = QString("%1.%2").arg(2 + (p->level >> 2)).arg(p->level & 3);
        else                                      lv = QString::number(p->level);
        s += (s.isEmpty() ? "" : " @ ") + QString("L%1").arg(lv);
    }
    return s;
}

QString codecName(AVCodecID id)
{
    const AVCodecDescriptor* d = avcodec_descriptor_get(id);
    if (!d) return {};
    return d->long_name ? QString("%1（%2）").arg(QString::fromUtf8(d->name).toUpper(), QString::fromUtf8(d->long_name))
                        : QString::fromUtf8(d->name).toUpper();
}

QString fourcc(uint32_t tag)
{
    if (!tag) return {};
    char buf[AV_FOURCC_MAX_STRING_SIZE] = {};
    av_fourcc_make_string(buf, tag);
    return QString::fromLatin1(buf).trimmed();
}

void videoRows(Rows& rows, const AVFormatContext* fc, const AVStream* st)
{
    const AVCodecParameters* p = st->codecpar;
    add(rows, "ID", QString::number(st->index));
    add(rows, "形式", codecName(p->codec_id));
    add(rows, "コーデック ID", fourcc(p->codec_tag));
    add(rows, "プロファイル", profileOf(p));
    add(rows, "幅",   p->width  > 0 ? QString("%L1 ピクセル").arg(p->width)  : QString());
    add(rows, "高さ", p->height > 0 ? QString("%L1 ピクセル").arg(p->height) : QString());

    // 表示アスペクト比（画素アスペクト比を考慮）
    if (p->width > 0 && p->height > 0) {
        AVRational sar = st->sample_aspect_ratio.num ? st->sample_aspect_ratio : p->sample_aspect_ratio;
        if (sar.num <= 0 || sar.den <= 0) sar = AVRational{1, 1};
        AVRational dar;
        av_reduce(&dar.num, &dar.den, int64_t(p->width) * sar.num, int64_t(p->height) * sar.den, 1024 * 1024);
        add(rows, "表示アスペクト比", QString("%1:%2（%3）").arg(dar.num).arg(dar.den)
                                       .arg(QString::number(av_q2d(dar), 'f', 3)));
        if (sar.num != sar.den) add(rows, "画素アスペクト比", QString("%1:%2").arg(sar.num).arg(sar.den));
    }

    // 回転（表示行列）
    if (const AVPacketSideData* sd = av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX)) {
        if (sd->size >= 9 * 4) {
            const double rot = av_display_rotation_get(reinterpret_cast<const int32_t*>(sd->data));
            if (!std::isnan(rot) && std::fabs(rot) > 0.01) add(rows, "回転", QString("%1°").arg(-rot, 0, 'f', 0));
        }
    }

    const QString avg = fmtRational(st->avg_frame_rate), real = fmtRational(st->r_frame_rate);
    add(rows, "フレームレート", avg.isEmpty() ? (real.isEmpty() ? QString() : real + " fps") : avg + " fps");
    if (!avg.isEmpty() && !real.isEmpty())
        add(rows, "フレームレートの種類", av_cmp_q(st->avg_frame_rate, st->r_frame_rate) == 0 ? "固定（CFR）" : "可変（VFR）");
    if (st->nb_frames > 0) add(rows, "フレーム数", QString("%L1").arg(st->nb_frames));
    add(rows, "長さ", fmtDuration(streamDuration(fc, st)));
    add(rows, "ビットレート", fmtBitrate(p->bit_rate));

    // 画素の形式とビット深度
    if (p->format >= 0) {
        const AVPixelFormat pf = AVPixelFormat(p->format);
        add(rows, "ピクセルフォーマット", QString::fromUtf8(av_get_pix_fmt_name(pf)));
        if (const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(pf)) {
            add(rows, "ビット深度", QString("%1 ビット").arg(d->comp[0].depth));
            if (!(d->flags & AV_PIX_FMT_FLAG_RGB) && d->nb_components >= 3) {
                const int cw = d->log2_chroma_w, ch = d->log2_chroma_h;
                add(rows, "クロマサブサンプリング", cw == 1 && ch == 1 ? "4:2:0" : cw == 1 && ch == 0 ? "4:2:2"
                                                 : cw == 0 && ch == 0 ? "4:4:4" : QString("%1x%2").arg(1 << cw).arg(1 << ch));
            }
        }
    }
    if (p->color_range != AVCOL_RANGE_UNSPECIFIED)
        add(rows, "色範囲", p->color_range == AVCOL_RANGE_JPEG ? "Full（0-255）" : "Limited（16-235）");
    if (p->color_primaries != AVCOL_PRI_UNSPECIFIED) add(rows, "色の三原色",   QString::fromUtf8(av_color_primaries_name(p->color_primaries)));
    if (p->color_trc != AVCOL_TRC_UNSPECIFIED)       add(rows, "伝達特性",     QString::fromUtf8(av_color_transfer_name(p->color_trc)));
    if (p->color_space != AVCOL_SPC_UNSPECIFIED)     add(rows, "行列係数",     QString::fromUtf8(av_color_space_name(p->color_space)));
    if (p->field_order > AV_FIELD_PROGRESSIVE)       add(rows, "走査方式", "インターレース");
    else if (p->field_order == AV_FIELD_PROGRESSIVE) add(rows, "走査方式", "プログレッシブ");
    if (p->video_delay > 0) add(rows, "B フレーム（並べ替えの遅延）", QString::number(p->video_delay));
    add(rows, "タイムベース", QString("%1/%2").arg(st->time_base.num).arg(st->time_base.den));
    if (tag(st->metadata, "language") != "und") add(rows, "言語", tag(st->metadata, "language"));
    addTags(rows, st->metadata, {"language"});
}

void audioRows(Rows& rows, const AVFormatContext* fc, const AVStream* st)
{
    const AVCodecParameters* p = st->codecpar;
    add(rows, "ID", QString::number(st->index));
    add(rows, "形式", codecName(p->codec_id));
    add(rows, "コーデック ID", fourcc(p->codec_tag));
    add(rows, "プロファイル", profileOf(p));
    if (p->ch_layout.nb_channels > 0) {
        add(rows, "チャンネル数", QString("%1 チャンネル").arg(p->ch_layout.nb_channels));
        char buf[128] = {};
        if (av_channel_layout_describe(&p->ch_layout, buf, sizeof(buf)) > 0) add(rows, "チャンネル配置", QString::fromUtf8(buf));
    }
    add(rows, "サンプリングレート", p->sample_rate > 0 ? QString("%L1 Hz").arg(p->sample_rate) : QString());
    if (p->format >= 0) add(rows, "サンプルフォーマット", QString::fromUtf8(av_get_sample_fmt_name(AVSampleFormat(p->format))));
    if (p->bits_per_raw_sample > 0) add(rows, "ビット深度", QString("%1 ビット").arg(p->bits_per_raw_sample));
    add(rows, "長さ", fmtDuration(streamDuration(fc, st)));
    add(rows, "ビットレート", fmtBitrate(p->bit_rate));
    if (tag(st->metadata, "language") != "und") add(rows, "言語", tag(st->metadata, "language"));
    addTags(rows, st->metadata, {"language"});
}

void otherRows(Rows& rows, const AVFormatContext* fc, const AVStream* st)
{
    const AVCodecParameters* p = st->codecpar;
    add(rows, "ID", QString::number(st->index));
    add(rows, "形式", codecName(p->codec_id));
    add(rows, "コーデック ID", fourcc(p->codec_tag));
    add(rows, "長さ", fmtDuration(streamDuration(fc, st)));
    if (tag(st->metadata, "language") != "und") add(rows, "言語", tag(st->metadata, "language"));
    addTags(rows, st->metadata, {"language"});
}

// コンテナ形式の表示名（FFmpeg の名前は mov,mp4,… のようにまとめてあるので拡張子で絞る）
QString containerName(const AVInputFormat* f, const QString& ext)
{
    if (!f) return {};
    const QString n = QString::fromUtf8(f->name);
    const QString e = ext.toLower();
    if (n.contains("mp4")) return e == "mov" ? "QuickTime（MOV）" : "MPEG-4（MP4）";
    if (n.contains("matroska")) return e == "webm" ? "WebM" : "Matroska（MKV）";
    if (n == "avi") return "AVI";
    return f->long_name ? QString::fromUtf8(f->long_name) : n;
}

// "isomiso2avc1mp41" → "isom / iso2 / avc1 / mp41"
QString splitBrands(const QString& s)
{
    QStringList out;
    for (int i = 0; i + 4 <= s.size(); i += 4) out << s.mid(i, 4);
    return out.join(" / ");
}

} // namespace

MediaProbe::MediaProbe(QObject* parent)
    : QObject(parent)
{
    connect(&watcher_, &QFutureWatcher<Result>::finished, this, [this] {
        const Result r = watcher_.result();
        sections_ = r.sections;
        error_    = r.error;
        busy_     = false;
        emit changed();
    });
}

MediaProbe::~MediaProbe()
{
    watcher_.waitForFinished();
}

void MediaProbe::open(const QString& path)
{
    if (busy_) return;
    fileName_ = QFileInfo(path).fileName();
    sections_.clear();
    error_.clear();
    busy_ = true;
    emit changed();
    emit opened();
    watcher_.setFuture(QtConcurrent::run(&MediaProbe::probe, path));
}

MediaProbe::Result MediaProbe::probe(const QString& path)
{
    Result r;
    AVFormatContext* fc = nullptr;
    const QByteArray p8 = path.toUtf8();
    if (avformat_open_input(&fc, p8.constData(), nullptr, nullptr) < 0) {
        r.error = "ファイルを開けませんでした。";
        return r;
    }
    avformat_find_stream_info(fc, nullptr);

    // ---- 全般 ----
    Rows g;
    const QFileInfo fi(path);
    add(g, "ファイル名", fi.fileName());
    add(g, "完全なパス", QDir::toNativeSeparators(fi.absoluteFilePath()));
    add(g, "形式", containerName(fc->iformat, fi.suffix()));
    add(g, "ブランド", tag(fc->metadata, "major_brand") + (tag(fc->metadata, "compatible_brands").isEmpty() ? QString()
                       : "（" + splitBrands(tag(fc->metadata, "compatible_brands")) + "）"));
    add(g, "サイズ", fmtSize(fi.size()));
    add(g, "長さ", fc->duration != AV_NOPTS_VALUE ? fmtDuration(fc->duration / double(AV_TIME_BASE)) : QString());
    add(g, "総ビットレート", fmtBitrate(fc->bit_rate));
    int nv = 0, na = 0, no = 0;
    for (unsigned i = 0; i < fc->nb_streams; ++i) {
        const auto t = fc->streams[i]->codecpar->codec_type;
        if (t == AVMEDIA_TYPE_VIDEO && !(fc->streams[i]->disposition & AV_DISPOSITION_ATTACHED_PIC)) ++nv;
        else if (t == AVMEDIA_TYPE_AUDIO) ++na;
        else ++no;
    }
    add(g, "ストリーム", QString("映像 %1 / 音声 %2 / その他 %3").arg(nv).arg(na).arg(no));
    add(g, "作成日時", tag(fc->metadata, "creation_time"));
    add(g, "使用したエンコーダ", tag(fc->metadata, "encoder"));
    if (fc->nb_chapters > 0) add(g, "チャプター", QString::number(fc->nb_chapters));
    addTags(g, fc->metadata, {"major_brand", "minor_version", "compatible_brands", "creation_time", "encoder"});
    r.sections << QVariantMap{ {"title", "全般"}, {"rows", g} };

    // ---- ストリームごと ----
    int iv = 0, ia = 0, io = 0;
    for (unsigned i = 0; i < fc->nb_streams; ++i) {
        const AVStream* st = fc->streams[i];
        Rows rows;
        QString title;
        switch (st->codecpar->codec_type) {
        case AVMEDIA_TYPE_VIDEO:
            if (st->disposition & AV_DISPOSITION_ATTACHED_PIC) { title = QString("カバー画像 #%1").arg(++io); otherRows(rows, fc, st); }
            else { title = QString("映像 #%1").arg(++iv); videoRows(rows, fc, st); }
            break;
        case AVMEDIA_TYPE_AUDIO:
            title = QString("音声 #%1").arg(++ia); audioRows(rows, fc, st); break;
        case AVMEDIA_TYPE_SUBTITLE:
            title = QString("字幕 #%1").arg(++io); otherRows(rows, fc, st); break;
        default:
            title = QString("その他 #%1").arg(++io); otherRows(rows, fc, st); break;
        }
        r.sections << QVariantMap{ {"title", title}, {"rows", rows} };
    }

    avformat_close_input(&fc);
    return r;
}

QString MediaProbe::toText() const
{
    // 項目名の幅をそろえる（全角は 2 文字分として数える）
    auto width = [](const QString& s) { int w = 0; for (const QChar c : s) w += c.unicode() >= 0x2E80 ? 2 : 1; return w; };
    int kw = 0;
    for (const QVariant& sec : sections_)
        for (const QVariant& row : sec.toMap().value("rows").toList())
            kw = std::max(kw, width(row.toMap().value("k").toString()));

    QString out;
    QTextStream ts(&out);
    for (const QVariant& sec : sections_) {
        const QVariantMap m = sec.toMap();
        ts << m.value("title").toString() << "\n";
        for (const QVariant& row : m.value("rows").toList()) {
            const QVariantMap rm = row.toMap();
            const QString k = rm.value("k").toString();
            ts << k << QString(kw - width(k) + 2, ' ') << ": " << rm.value("v").toString() << "\n";
        }
        ts << "\n";
    }
    return out;
}

void MediaProbe::copyToClipboard() const
{
    if (QClipboard* cb = QGuiApplication::clipboard()) cb->setText(toText());
}

void MediaProbe::saveAs() const
{
    const QString base = QFileInfo(fileName_).completeBaseName();
    const QString path = QFileDialog::getSaveFileName(nullptr, tr("名前を付けて保存"), base + "_info.txt",
                                                      tr("テキストファイル (*.txt)"));
    if (path.isEmpty()) return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) return;
    f.write(toText().toUtf8());
}
