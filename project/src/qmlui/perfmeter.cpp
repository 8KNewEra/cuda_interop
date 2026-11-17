#include "src/qmlui/perfmeter.h"

#include <QOpenGLFunctions_4_5_Core>
#include <QVariantList>
#include <algorithm>

QVariantMap PerfMeter::Acc::toMap() const
{
    // n == 0 のときは avg / max を -1（HUD は「—」を出す）
    QVariantMap m;
    m["avg"] = n ? sum / n : -1.0;
    m["max"] = n ? max : -1.0;
    m["n"]   = n;
    return m;
}

void PerfMeter::setEnabled(bool on)
{
    if (on_ == on) return;
    on_ = on;
    // 切り替えたら集計をやり直す（OFF の間の古い値を混ぜない）
    for (Slot& s : slots_) { s.used = false; for (bool& b : s.mark) b = false; s.cuda = false; }
    cur_ = -1;
    haveLast_ = false;
    decode_.clear(); convert_.clear(); filter_.clear(); display_.clear(); hist_.clear(); cpu_.clear(); interval_ms_.clear();
    transfer_.clear(); transferGbps_.clear(); ratio_.clear(); compress_.clear(); expand_.clear();
    fetch_.clear(); decomp_.clear(); inv_.clear();
    intervals_.clear();
    parts_.clear();
}

void PerfMeter::create(QOpenGLFunctions_4_5_Core* gl)
{
    if (created_) return;
    for (Slot& s : slots_) {
        gl->glGenQueries(kGlMarks, s.query);
        for (cudaEvent_t& e : s.ev) cudaEventCreate(&e);   // 時間を測るので DisableTiming は付けない
    }
    created_ = true;
}

void PerfMeter::destroy(QOpenGLFunctions_4_5_Core* gl)
{
    if (!created_) return;
    for (Slot& s : slots_) {
        if (gl) gl->glDeleteQueries(kGlMarks, s.query);
        for (cudaEvent_t& e : s.ev) { if (e) cudaEventDestroy(e); e = nullptr; }
    }
    created_ = false;
}

void PerfMeter::frameBegin(QOpenGLFunctions_4_5_Core* gl)
{
    cur_ = -1;
    if (!on_) return;
    create(gl);
    collect(gl);

    // フレーム間隔（新しいフレームを受け取った間隔）
    if (haveLast_) {
        const double ms = interval_.nsecsElapsed() / 1e6;
        interval_ms_.add(ms);
        intervals_.push_back(float(ms));
    }
    interval_.restart();
    haveLast_ = true;
    frameCpu_.restart();

    // 空いている slot を使う。全部が結果待ちならこのフレームは GPU 側を測らない（待たない）
    Slot& s = slots_[next_];
    if (s.used) return;
    cur_ = next_;
    next_ = (next_ + 1) % kRing;
    s.used = true;
    for (bool& b : s.mark) b = false;
    s.cuda = false;
}

void PerfMeter::convertBegin(cudaStream_t st)
{
    if (!on_ || cur_ < 0) return;
    cudaEventRecord(slots_[cur_].ev[0], st);
}

void PerfMeter::convertEnd(cudaStream_t st)
{
    if (!on_ || cur_ < 0) return;
    cudaEventRecord(slots_[cur_].ev[1], st);
    slots_[cur_].cuda = true;
}

void PerfMeter::glMark(QOpenGLFunctions_4_5_Core* gl, GlMark m)
{
    if (!on_ || cur_ < 0) return;
    gl->glQueryCounter(slots_[cur_].query[m], GL_TIMESTAMP);
    slots_[cur_].mark[m] = true;
}

void PerfMeter::addHistogram(double ms)
{
    if (on_) hist_.add(ms);
}

void PerfMeter::addDecode(double ms)
{
    if (on_) decode_.add(ms);
}

void PerfMeter::addTransfer(double ms, double gbps)
{
    if (!on_) return;
    transfer_.add(ms);
    transferGbps_.add(gbps);
}

void PerfMeter::addTransferCodec(double ratio, double compressMs, double expandMs)
{
    if (!on_) return;
    ratio_.add(ratio);
    compress_.add(compressMs);
    expand_.add(expandMs);
}

void PerfMeter::addExpandParts(double fetchMs, double decompMs, double invMs)
{
    if (!on_) return;
    fetch_.add(fetchMs); decomp_.add(decompMs); inv_.add(invMs);
}

void PerfMeter::addDecodeBreakdown(const QVariantList& parts)
{
    if (!on_) return;
    for (const QVariant& v : parts) {
        const QVariantMap p = v.toMap();
        const QString key = p.value("key").toString();
        auto it = std::find_if(parts_.begin(), parts_.end(), [&](const Part& x) { return x.key == key; });
        if (it == parts_.end()) { parts_.push_back({ key, p.value("label").toString(), {} }); it = parts_.end() - 1; }
        it->acc.add(p.value("ms").toDouble());
    }
}

void PerfMeter::frameEnd()
{
    if (!on_) return;
    cpu_.add(frameCpu_.nsecsElapsed() / 1e6);
    cur_ = -1;
}

// 結果が届いた slot を集計する（待たない。届いていなければ次の機会に）
void PerfMeter::collect(QOpenGLFunctions_4_5_Core* gl)
{
    for (Slot& s : slots_) {
        if (!s.used) continue;
        // CUDA（変換）
        if (s.cuda && cudaEventQuery(s.ev[1]) != cudaSuccess) continue;
        // GL: 最後に打った点が届いていれば全部届いている（GPU は順に進む）
        int last = -1;
        for (int m = 0; m < kGlMarks; ++m) if (s.mark[m]) last = m;
        if (last >= 0) {
            GLint avail = 0;
            gl->glGetQueryObjectiv(s.query[last], GL_QUERY_RESULT_AVAILABLE, &avail);
            if (!avail) continue;
        }
        if (s.cuda) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, s.ev[0], s.ev[1]) == cudaSuccess) convert_.add(ms);
            else cudaGetLastError();
        }
        auto stamp = [&](int m) { GLuint64 t = 0; gl->glGetQueryObjectui64v(s.query[m], GL_QUERY_RESULT, &t); return t; };
        if (s.mark[FilterBegin] && s.mark[FilterEnd])
            filter_.add((stamp(FilterEnd) - stamp(FilterBegin)) / 1e6);
        if (s.mark[DisplayBegin] && s.mark[DisplayEnd])
            display_.add((stamp(DisplayEnd) - stamp(DisplayBegin)) / 1e6);
        s.used = false;
    }
}

QVariantMap PerfMeter::take()
{
    QVariantMap m;
    m["decode"]   = decode_.toMap();
    m["convert"]  = convert_.toMap();
    m["filter"]   = filter_.toMap();
    m["display"]  = display_.toMap();
    m["hist"]     = hist_.toMap();
    m["cpu"]      = cpu_.toMap();
    m["interval"] = interval_ms_.toMap();
    m["transfer"] = transfer_.toMap();
    m["transferGbps"] = transferGbps_.n ? transferGbps_.sum / transferGbps_.n : -1.0;   // 平均の転送速度（GB/s）
    m["xferRatio"]    = ratio_.n ? ratio_.sum / ratio_.n : -1.0;                     // 圧縮率（元 ÷ 圧縮後。圧縮転送のときだけ）
    m["compress"]     = compress_.toMap();                                            // 別 GPU での前処理と圧縮
    m["expand"]       = expand_.toMap();
    QVariantList dp;                                                                  // デコード（待ち）の内訳
    for (Part& p : parts_) {
        QVariantMap e = p.acc.toMap();
        e["label"] = p.label;
        dp << e;
    }
    m["decodeParts"] = dp;
    parts_.clear();
    QVariantList ep;                                                                  // 展開の内訳
    if (fetch_.n > 0) {
        auto part = [&](const Acc& a, const char* label) { QVariantMap e = a.toMap(); e["label"] = QString::fromUtf8(label); ep << e; };
        part(fetch_,  "取り込み");
        part(decomp_, "nvCOMP 展開");
        part(inv_,    "差分戻し・書き込み");
    }
    m["expandParts"] = ep;
    fetch_.clear(); decomp_.clear(); inv_.clear();                                              // 表示用 GPU での取り込み・展開
    QVariantList iv;
    iv.reserve(int(intervals_.size()));
    for (float v : intervals_) iv << double(v);
    m["intervals"] = iv;
    decode_.clear(); convert_.clear(); filter_.clear(); display_.clear(); hist_.clear(); cpu_.clear(); interval_ms_.clear();
    transfer_.clear(); transferGbps_.clear(); ratio_.clear(); compress_.clear(); expand_.clear();
    intervals_.clear();
    return m;
}
