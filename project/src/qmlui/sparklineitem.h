#ifndef SPARKLINEITEM_H
#define SPARKLINEITEM_H

// ==============================================================
// SparklineItem（推移グラフ。qml/Sparkline.qml の中身）
//
//   以前は QML の Canvas（Image ＋ Immediate）で、描き直すたびに GUI スレッドで JS を回し、
//   QPainter でソフトウェア描画してテクスチャを上げ直していた。render loop が basic なので、
//   それが映像の upload と swap の間に割り込んで表示の間隔を乱していた（性能 HUD の 250ms 周期）。
//   ここではシーングラフのジオメトリ（三角形）を作るだけにする。点は高々数百なので CPU は 0.1ms 未満。
//
//   見た目は Canvas 版と同じ
//     ・背景 = bgColor を bgOpacity で塗る
//     ・refValue >= 0 なら基準線（破線 3px / 3px、refColor）
//     ・values（古い → 新しい。右端が最新、横軸は capacity 点）の線（lineWidth）と下の塗り（fillOpacity）
//     ・null / undefined は欠け（線と塗りを切る）。1 点だけの区間は短い横線
//   線の縁は 1px の透明な縁取り（頂点の色で 0 へ落とす）でぼかす（MSAA なしでもギザギザにしない）
// ==============================================================

#include <QQuickItem>
#include <QColor>
#include <QVariantList>
#include <vector>

class SparklineItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QVariantList values    READ values    WRITE setValues    NOTIFY valuesChanged)
    Q_PROPERTY(int    capacity        READ capacity  WRITE setCapacity  NOTIFY styleChanged)
    Q_PROPERTY(qreal  maxValue        READ maxValue  WRITE setMaxValue  NOTIFY styleChanged)
    Q_PROPERTY(qreal  refValue        READ refValue  WRITE setRefValue  NOTIFY styleChanged)
    Q_PROPERTY(QColor lineColor       READ lineColor WRITE setLineColor NOTIFY styleChanged)
    Q_PROPERTY(QColor refColor        READ refColor  WRITE setRefColor  NOTIFY styleChanged)
    Q_PROPERTY(QColor bgColor         READ bgColor   WRITE setBgColor   NOTIFY styleChanged)
    Q_PROPERTY(qreal  bgOpacity       READ bgOpacity WRITE setBgOpacity NOTIFY styleChanged)
    Q_PROPERTY(qreal  fillOpacity     READ fillOpacity WRITE setFillOpacity NOTIFY styleChanged)
    Q_PROPERTY(qreal  lineWidth       READ lineWidth WRITE setLineWidth NOTIFY styleChanged)

public:
    explicit SparklineItem(QQuickItem* parent = nullptr);

    QVariantList values() const { return valuesVar_; }
    void setValues(const QVariantList& v);
    int    capacity()   const { return capacity_; }
    void   setCapacity(int c);
    qreal  maxValue()   const { return maxValue_; }
    void   setMaxValue(qreal v);
    qreal  refValue()   const { return refValue_; }
    void   setRefValue(qreal v);
    QColor lineColor()  const { return lineColor_; }
    void   setLineColor(const QColor& c);
    QColor refColor()   const { return refColor_; }
    void   setRefColor(const QColor& c);
    QColor bgColor()    const { return bgColor_; }
    void   setBgColor(const QColor& c);
    qreal  bgOpacity()  const { return bgOpacity_; }
    void   setBgOpacity(qreal v);
    qreal  fillOpacity() const { return fillOpacity_; }
    void   setFillOpacity(qreal v);
    qreal  lineWidth()  const { return lineWidth_; }
    void   setLineWidth(qreal v);

signals:
    void valuesChanged();
    void styleChanged();

protected:
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) override;
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;

private:
    void changed() { update(); }

    QVariantList        valuesVar_;   // QML から来たまま（読み返し用）
    std::vector<double> values_;      // NaN = 欠け
    int    capacity_    = 60;
    qreal  maxValue_    = 100;
    qreal  refValue_    = -1;
    QColor lineColor_   = QColor("#4FD1C5");
    QColor refColor_    = QColor("#8F97A2");
    QColor bgColor_     = QColor("#2A2E35");
    qreal  bgOpacity_   = 0.45;
    qreal  fillOpacity_ = 0.18;
    qreal  lineWidth_   = 1.5;
};

#endif // SPARKLINEITEM_H
