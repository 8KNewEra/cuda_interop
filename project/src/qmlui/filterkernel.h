#ifndef FILTERKERNEL_H
#define FILTERKERNEL_H

#include <QObject>
#include <QVariantList>
#include <QVector>

// 畳み込みフィルタ 1 本ぶんの係数（ガウシアン / 平均化）
//   ・サイズ（3/5/7/9）と σ（ガウシアンのみ）から自動生成
//   ・セルごとに直接書き換え可（custom = true）。直接編集した係数は ini に保存しない
//   ・normalize ON ならシェーダには合計で割った値を渡す
//   ・サイズ / σ の変更・リセットでパラメータから作り直す
//   変更は changed() → PlayerController がシェーダへ送る（VideoEngine::setFilterKernel）
class FilterKernel : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString kind      READ kind      CONSTANT)          // "gaussian" | "box"
    Q_PROPERTY(int     size      READ size      NOTIFY changed)    // 一辺（奇数）
    Q_PROPERTY(int     sigma100  READ sigma100  NOTIFY changed)    // σ × 100（NumberSpin 用）
    Q_PROPERTY(bool    normalize READ normalize NOTIFY changed)
    Q_PROPERTY(bool    custom    READ custom    NOTIFY changed)    // セルを直接編集したか
    Q_PROPERTY(QVariantList coeffs READ coeffs  NOTIFY changed)    // 入力値（行優先 size×size）
    Q_PROPERTY(double  sum       READ sum       NOTIFY changed)    // 入力値の合計
    Q_PROPERTY(QVariantList sizeOptions READ sizeOptions CONSTANT)

public:
    static constexpr int kMaxSize = 9;      // シェーダの配列（81 要素）に合わせる

    enum class Kind { Gaussian, Box };
    explicit FilterKernel(Kind kind, QObject* parent = nullptr);

    QString kind() const   { return kind_ == Kind::Gaussian ? "gaussian" : "box"; }
    int  size() const      { return size_; }
    int  sigma100() const  { return sigma100_; }
    bool normalize() const { return normalize_; }
    bool custom() const    { return custom_; }
    QVariantList coeffs() const;
    double sum() const;
    QVariantList sizeOptions() const;

    // シェーダへ渡す係数（正規化済み、行優先 size×size）
    QVector<float> effective() const;

    Q_INVOKABLE void setSize(int n);
    Q_INVOKABLE void setSigma100(int v);
    Q_INVOKABLE void setNormalize(bool on);
    Q_INVOKABLE void setCoeff(int index, double value);
    Q_INVOKABLE void reset();          // パラメータから作り直す（直接編集は捨てる）

signals:
    void changed();

private:
    Kind kind_;
    int  size_      = 7;
    int  sigma100_  = 84;      // 既定は旧 7x7 固定カーネルに近い σ ≈ 0.84
    bool normalize_ = true;
    bool custom_    = false;
    QVector<double> k_;        // 入力値

    void generate();
    void saveToSettings() const;   // サイズ・σ・正規化を g_AppSettings へ（終了時に ini 保存）
};

#endif // FILTERKERNEL_H
