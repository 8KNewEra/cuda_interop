#ifndef TONECURVE_H
#define TONECURVE_H

#include <QString>
#include <QVector>
#include <QPoint>
#include <array>

// トーンカーブの点・文字列・LUT の変換（UI と描画で共用する純粋関数）
//   点は 0..255 の整数（x = 入力、y = 出力）。両端（x = 0 / 255）を必ず持つ
//   点の間は単調 3 次補間（Fritsch–Carlson）。点の間で行き過ぎて波打たない
namespace ToneCurve {

constexpr int kMaxPoints = 16;

// 直線（初期状態）
QVector<QPoint> identity();
bool isIdentity(const QVector<QPoint>& pts);

// ini の文字列 "0:0,64:52,255:255" ⇔ 点
//   壊れた文字列は直線。範囲外は 0..255 に収め、x で並べ替え、同じ x はまとめ、
//   両端がなければ足し、最大 kMaxPoints 個にする
QVector<QPoint> parse(const QString& s);
QString serialize(const QVector<QPoint>& pts);
QVector<QPoint> normalize(QVector<QPoint> pts);

// 256 段の対応表（0..1）
std::array<float, 256> lut(const QVector<QPoint>& pts);

} // namespace ToneCurve

#endif // TONECURVE_H
