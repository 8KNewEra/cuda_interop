#ifndef VIDEOITEM_H
#define VIDEOITEM_H

// ==============================================================
// VideoItem / HistogramItem
//
//   VideoEngine が描いた FBO テクスチャを、コピーせずにシーングラフへ
//   載せるだけの QQuickItem。
//
//   ・テクスチャは QNativeInterface::QSGOpenGLTexture::fromNative で包む。
//     GL の実体は engine が持ち続ける（こちらは削除しない）。
//   ・FBO は GL の慣例どおり下の行から並んでいるので、表示時に上下反転する。
//   ・ズーム／パンは GLWidget から移植。左/中ドラッグ=移動、
//     ホイール=拡大縮小（カーソル位置固定）、ダブルクリック=等倍。
// ==============================================================

#include <QQuickItem>
#include <QRectF>
#include <QSize>
#include <QPointF>

class VideoItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(bool   hasVideo    READ hasVideo    NOTIFY frameChanged)
    Q_PROPERTY(QSize  sourceSize  READ sourceSize  NOTIFY frameChanged)
    Q_PROPERTY(double zoom        READ zoom        NOTIFY viewChanged)
    Q_PROPERTY(QRectF videoRect   READ videoRect   NOTIFY viewChanged)  // 表示枠（item 座標）
    Q_PROPERTY(QRectF viewRect    READ viewRect    NOTIFY viewChanged)  // 表示中の範囲（元画像に対する 0..1）
    Q_PROPERTY(bool   interactive READ interactive WRITE setInteractive NOTIFY interactiveChanged)

public:
    explicit VideoItem(QQuickItem* parent = nullptr);

    bool   hasVideo() const;
    QSize  sourceSize() const;
    double zoom() const { return zoomFactor; }
    QRectF videoRect() const;
    QRectF viewRect() const;
    bool   interactive() const { return interactive_; }
    void   setInteractive(bool on);

    Q_INVOKABLE void resetZoom();

signals:
    void frameChanged();
    void viewChanged();
    void interactiveChanged();

protected:
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) override;
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;

    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void onFrameReady();
    double zoomMax() const;
    void   clampView();
    void   visibleSrcRect(double& sx, double& sy, double& sw, double& sh) const;
    QPointF itemToDisplayUV(const QPointF& pos) const;

    // ズーム（viewCx/viewCy は元画像座標の表示中心、左上原点）
    double zoomFactor = 1.0;
    double viewCx = 0.0;
    double viewCy = 0.0;
    bool   panning = false;
    Qt::MouseButton panButton = Qt::NoButton;
    QPointF panLastPos;
    bool interactive_ = true;

    QSize  lastSize_;           // 解像度が変わったらズームを戻す

    // レンダースレッド側（basic loop なので実際は GUI スレッド）
    unsigned int texId_ = 0;
    QSize        texSize_;
};

class HistogramItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(bool valid READ valid NOTIFY changed)

public:
    explicit HistogramItem(QQuickItem* parent = nullptr);
    bool valid() const;

signals:
    void changed();

protected:
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) override;
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;
    void itemChange(ItemChange change, const ItemChangeData& value) override;

private:
    void pushViewport();
    unsigned int texId_ = 0;
    QSize        texSize_;
};

#endif // VIDEOITEM_H
