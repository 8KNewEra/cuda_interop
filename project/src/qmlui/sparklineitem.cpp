#include "src/qmlui/sparklineitem.h"

#include <QSGGeometryNode>
#include <QSGVertexColorMaterial>
#include <algorithm>
#include <cmath>
#include <limits>

SparklineItem::SparklineItem(QQuickItem* parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true);
}

// ---- プロパティ ----
void SparklineItem::setValues(const QVariantList& v)
{
    // 推移は呼び出し側が毎回作り直した配列を渡す（中身が同じでも描き直す。点は数百なので安い）
    valuesVar_ = v;
    values_.resize(size_t(v.size()));
    for (int i = 0; i < v.size(); ++i) {
        const QVariant& e = v.at(i);
        double d = std::numeric_limits<double>::quiet_NaN();
        if (e.isValid() && !e.isNull()) {
            bool ok = false;
            const double x = e.toDouble(&ok);
            if (ok && std::isfinite(x)) d = x;
        }
        values_[size_t(i)] = d;
    }
    emit valuesChanged();
    changed();
}

#define SPARK_SETTER(name, member, type)                         \
    void SparklineItem::name(type v)                             \
    {                                                            \
        if (member == v) return;                                 \
        member = v;                                              \
        emit styleChanged();                                     \
        changed();                                               \
    }
SPARK_SETTER(setCapacity,    capacity_,    int)
SPARK_SETTER(setMaxValue,    maxValue_,    qreal)
SPARK_SETTER(setRefValue,    refValue_,    qreal)
SPARK_SETTER(setLineColor,   lineColor_,   const QColor&)
SPARK_SETTER(setRefColor,    refColor_,    const QColor&)
SPARK_SETTER(setBgColor,     bgColor_,     const QColor&)
SPARK_SETTER(setBgOpacity,   bgOpacity_,   qreal)
SPARK_SETTER(setFillOpacity, fillOpacity_, qreal)
SPARK_SETTER(setLineWidth,   lineWidth_,   qreal)
#undef SPARK_SETTER

void SparklineItem::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry)
{
    QQuickItem::geometryChange(newGeometry, oldGeometry);
    if (newGeometry.size() != oldGeometry.size()) update();
}

// ---- 描画（三角形を並べるだけ。QSGVertexColorMaterial、色は premultiplied）----
namespace {

struct Builder {
    std::vector<QSGGeometry::ColoredPoint2D> v;

    static QSGGeometry::ColoredPoint2D pt(float x, float y, const QColor& c, qreal alpha)
    {
        const float a = float(std::clamp(c.alphaF() * alpha, 0.0, 1.0));
        QSGGeometry::ColoredPoint2D p;
        p.set(x, y,
              uchar(std::lround(c.redF()   * a * 255)),
              uchar(std::lround(c.greenF() * a * 255)),
              uchar(std::lround(c.blueF()  * a * 255)),
              uchar(std::lround(a * 255)));
        return p;
    }
    // 四角形 a-b-c-d（周回順）。頂点ごとの不透明度 aa..ad
    void quad(QPointF a, QPointF b, QPointF c, QPointF d, const QColor& col,
              qreal aa = 1, qreal ab = 1, qreal ac = 1, qreal ad = 1)
    {
        const auto A = pt(float(a.x()), float(a.y()), col, aa);
        const auto B = pt(float(b.x()), float(b.y()), col, ab);
        const auto C = pt(float(c.x()), float(c.y()), col, ac);
        const auto D = pt(float(d.x()), float(d.y()), col, ad);
        v.push_back(A); v.push_back(B); v.push_back(C);
        v.push_back(A); v.push_back(C); v.push_back(D);
    }
    // 太さ w の線分（端は w/2 延ばして継ぎ目を埋める）＋ 両側に 1px の縁取り（透明へ落とす）
    void segment(QPointF p0, QPointF p1, qreal w, const QColor& col)
    {
        QPointF d = p1 - p0;
        const qreal len = std::hypot(d.x(), d.y());
        if (len < 1e-6) d = QPointF(1, 0); else d /= len;
        const QPointF n(-d.y(), d.x());
        const qreal h = w * 0.5, f = 1.0;
        const QPointF a = p0 - d * h, b = p1 + d * h;
        quad(a + n * h, b + n * h, b - n * h, a - n * h, col);                         // 芯
        quad(a + n * h, b + n * h, b + n * (h + f), a + n * (h + f), col, 1, 1, 0, 0); // 縁（上）
        quad(a - n * h, b - n * h, b - n * (h + f), a - n * (h + f), col, 1, 1, 0, 0); // 縁（下）
    }
};

} // namespace

QSGNode* SparklineItem::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*)
{
    const qreal w = width(), h = height();
    if (w <= 0 || h <= 0) { delete oldNode; return nullptr; }

    auto* node = static_cast<QSGGeometryNode*>(oldNode);
    if (!node) {
        node = new QSGGeometryNode;
        auto* geo = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geo->setDrawingMode(QSGGeometry::DrawTriangles);
        node->setGeometry(geo);
        node->setFlag(QSGNode::OwnsGeometry);
        node->setMaterial(new QSGVertexColorMaterial);
        node->setFlag(QSGNode::OwnsMaterial);
    }

    Builder b;
    b.v.reserve(6 + 6 * 64 + values_.size() * 24);

    // 背景
    b.quad({0, 0}, {w, 0}, {w, h}, {0, h}, bgColor_, bgOpacity_, bgOpacity_, bgOpacity_, bgOpacity_);

    const qreal mx = maxValue_ > 0 ? maxValue_ : 1;
    auto yOf = [&](double v) { return h - 1 - std::clamp(v / mx, 0.0, 1.0) * (h - 2); };

    // 基準線（破線 3px / 3px、太さ 1px。画素の境目にそろえて滲ませない）
    if (refValue_ >= 0) {
        const qreal y = std::floor(yOf(refValue_)) + 0.0;
        for (qreal x = 0; x < w; x += 6) {
            const qreal x1 = std::min(w, x + 3);
            b.quad({x, y}, {x1, y}, {x1, y + 1}, {x, y + 1}, refColor_);
        }
    }

    // 推移（右端が最新）
    const int n = int(values_.size());
    if (n > 0) {
        const qreal step = w / std::max(1, capacity_ - 1);
        const qreal x0 = w - (n - 1) * step;
        std::vector<QPointF> seg;
        seg.reserve(size_t(n));
        auto flush = [&] {
            if (seg.empty()) return;
            // 塗り（線の下から底まで）
            for (size_t i = 0; i + 1 < seg.size(); ++i) {
                const QPointF& p = seg[i];
                const QPointF& q = seg[i + 1];
                b.quad(p, q, {q.x(), h}, {p.x(), h}, lineColor_, fillOpacity_, fillOpacity_, fillOpacity_, fillOpacity_);
            }
            // 線
            if (seg.size() == 1) b.segment(seg[0], seg[0] + QPointF(1, 0), lineWidth_, lineColor_);
            for (size_t i = 0; i + 1 < seg.size(); ++i) b.segment(seg[i], seg[i + 1], lineWidth_, lineColor_);
            seg.clear();
        };
        for (int i = 0; i < n; ++i) {
            const double v = values_[size_t(i)];
            if (std::isnan(v)) { flush(); continue; }
            seg.emplace_back(x0 + i * step, yOf(v));
        }
        flush();
    }

    QSGGeometry* geo = node->geometry();
    geo->allocate(int(b.v.size()));
    std::copy(b.v.begin(), b.v.end(), geo->vertexDataAsColoredPoint2D());
    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}
