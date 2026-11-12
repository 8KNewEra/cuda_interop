#include "src/qmlui/filterkernel.h"
#include "src/main/__global__.h"

#include <QVariantMap>
#include <algorithm>
#include <cmath>

FilterKernel::FilterKernel(Kind kind, QObject* parent)
    : QObject(parent), kind_(kind)
{
    // 前回のサイズ・σ・正規化を復元（ini の値は範囲に収める）
    const AppSettings& a = g_AppSettings;
    if (kind_ == Kind::Gaussian) {
        size_      = std::clamp(a.gaussian_kernel_size | 1, 3, kMaxSize);
        sigma100_  = std::clamp(a.gaussian_sigma100, 10, 1000);
        normalize_ = a.gaussian_normalize;
    } else {
        size_      = std::clamp(a.averaging_kernel_size | 1, 3, kMaxSize);
        normalize_ = a.averaging_normalize;
    }
    generate();
    saveToSettings();   // 正規化した値を戻しておく
}

void FilterKernel::saveToSettings() const
{
    AppSettings& a = g_AppSettings;
    if (kind_ == Kind::Gaussian) {
        a.gaussian_kernel_size = size_;
        a.gaussian_sigma100    = sigma100_;
        a.gaussian_normalize   = normalize_;
    } else {
        a.averaging_kernel_size = size_;
        a.averaging_normalize   = normalize_;
    }
}

QVariantList FilterKernel::sizeOptions() const
{
    QVariantList list;
    for (int n = 3; n <= kMaxSize; n += 2)
        list << QVariantMap{ {"text", QString("%1×%1").arg(n)}, {"value", n} };
    return list;
}

QVariantList FilterKernel::coeffs() const
{
    QVariantList list;
    list.reserve(k_.size());
    for (double v : k_) list << v;
    return list;
}

double FilterKernel::sum() const
{
    double s = 0.0;
    for (double v : k_) s += v;
    return s;
}

QVector<float> FilterKernel::effective() const
{
    // 合計が 0 付近（エッジ検出のような係数）なら割らない
    const double s = sum();
    const double div = (normalize_ && std::abs(s) > 1e-9) ? s : 1.0;
    QVector<float> out(k_.size());
    for (int i = 0; i < k_.size(); ++i) out[i] = float(k_[i] / div);
    return out;
}

// 自動生成: ガウシアンは exp(-(x²+y²)/2σ²)、平均化はすべて同じ値
//   どちらも合計 1 にしてから入れる（セルに見える値 = 実際に掛かる重み）
void FilterKernel::generate()
{
    const int n = size_, r = n / 2;
    k_.resize(n * n);
    if (kind_ == Kind::Gaussian) {
        const double s = std::max(0.05, sigma100_ / 100.0);
        double total = 0.0;
        for (int y = -r; y <= r; ++y)
            for (int x = -r; x <= r; ++x)
                total += (k_[(y + r) * n + (x + r)] = std::exp(-(x * x + y * y) / (2.0 * s * s)));
        for (double& v : k_) v /= total;
    } else {
        std::fill(k_.begin(), k_.end(), 1.0 / (n * n));
    }
    custom_ = false;
}

void FilterKernel::setSize(int n)
{
    n = std::clamp(n | 1, 3, kMaxSize);      // 奇数に丸める
    if (n == size_) return;
    size_ = n;
    generate();
    saveToSettings();
    emit changed();
}

void FilterKernel::setSigma100(int v)
{
    v = std::clamp(v, 10, 1000);             // σ 0.10 〜 10.00
    if (v == sigma100_) return;
    sigma100_ = v;
    generate();
    saveToSettings();
    emit changed();
}

void FilterKernel::setNormalize(bool on)
{
    if (on == normalize_) return;
    normalize_ = on;
    saveToSettings();
    emit changed();
}

void FilterKernel::setCoeff(int index, double value)
{
    if (index < 0 || index >= k_.size() || !std::isfinite(value)) return;
    if (k_[index] == value) return;
    k_[index] = value;
    custom_ = true;
    emit changed();
}

void FilterKernel::reset()
{
    generate();
    emit changed();
}
