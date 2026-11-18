#include "src/qmlui/tonecurve.h"

#include <QStringList>
#include <algorithm>
#include <cmath>
#include <vector>

namespace ToneCurve {

QVector<QPoint> identity()
{
    return { QPoint(0, 0), QPoint(255, 255) };
}

bool isIdentity(const QVector<QPoint>& pts)
{
    // すべての点が対角線上なら直線と同じ
    for (const QPoint& p : pts)
        if (p.x() != p.y()) return false;
    return true;
}

QVector<QPoint> normalize(QVector<QPoint> pts)
{
    for (QPoint& p : pts) {
        p.setX(std::clamp(p.x(), 0, 255));
        p.setY(std::clamp(p.y(), 0, 255));
    }
    std::stable_sort(pts.begin(), pts.end(), [](const QPoint& a, const QPoint& b) { return a.x() < b.x(); });
    // 同じ x は最後のものを残す
    QVector<QPoint> out;
    for (const QPoint& p : pts) {
        if (!out.isEmpty() && out.last().x() == p.x()) out.last() = p;
        else out.push_back(p);
    }
    if (out.isEmpty() || out.first().x() != 0) out.prepend(QPoint(0, 0));
    if (out.last().x() != 255)                 out.append(QPoint(255, 255));
    // 多すぎる場合は両端を残して間を間引く
    while (out.size() > kMaxPoints) out.remove(out.size() / 2);
    return out;
}

QVector<QPoint> parse(const QString& s)
{
    QVector<QPoint> pts;
    for (const QString& item : s.split(',', Qt::SkipEmptyParts)) {
        const QStringList xy = item.trimmed().split(':');
        if (xy.size() != 2) return identity();
        bool okx = false, oky = false;
        const int x = xy[0].trimmed().toInt(&okx);
        const int y = xy[1].trimmed().toInt(&oky);
        if (!okx || !oky) return identity();
        pts.push_back(QPoint(x, y));
    }
    if (pts.isEmpty()) return identity();
    return normalize(pts);
}

QString serialize(const QVector<QPoint>& pts)
{
    QStringList list;
    for (const QPoint& p : pts) list << QString("%1:%2").arg(p.x()).arg(p.y());
    return list.join(',');
}

std::array<float, 256> lut(const QVector<QPoint>& in)
{
    const QVector<QPoint> pts = normalize(in);
    const int n = pts.size();
    std::array<float, 256> out{};

    // 区間ごとの傾き → 点ごとの接線（Fritsch–Carlson で単調性を保つ）
    std::vector<double> x(n), y(n), d(n - 1), m(n);
    for (int i = 0; i < n; ++i) { x[i] = pts[i].x(); y[i] = pts[i].y(); }
    for (int i = 0; i < n - 1; ++i) d[i] = (y[i + 1] - y[i]) / std::max(1.0, x[i + 1] - x[i]);
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (int i = 1; i < n - 1; ++i)
        m[i] = (d[i - 1] * d[i] <= 0) ? 0.0 : (d[i - 1] + d[i]) * 0.5;
    for (int i = 0; i < n - 1; ++i) {
        if (d[i] == 0) { m[i] = m[i + 1] = 0; continue; }
        const double a = m[i] / d[i], b = m[i + 1] / d[i];
        const double s = a * a + b * b;
        if (s > 9.0) {
            const double t = 3.0 / std::sqrt(s);
            m[i] = t * a * d[i];
            m[i + 1] = t * b * d[i];
        }
    }

    int seg = 0;
    for (int v = 0; v < 256; ++v) {
        while (seg < n - 2 && v > x[seg + 1]) ++seg;
        const double h = std::max(1.0, x[seg + 1] - x[seg]);
        const double t = std::clamp((v - x[seg]) / h, 0.0, 1.0);
        const double t2 = t * t, t3 = t2 * t;
        const double val = (2 * t3 - 3 * t2 + 1) * y[seg] + (t3 - 2 * t2 + t) * h * m[seg]
                         + (-2 * t3 + 3 * t2) * y[seg + 1] + (t3 - t2) * h * m[seg + 1];
        out[v] = float(std::clamp(val, 0.0, 255.0) / 255.0);
    }
    return out;
}

} // namespace ToneCurve
