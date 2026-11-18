#include "src/qmlui/videoitem.h"
#include "src/qmlui/videoengine.h"

#include <QQuickWindow>
#include <QSGImageNode>
#include <QSGTexture>
#include <QtQuick/qsgtexture_platform.h>
#include <QWheelEvent>
#include <QMouseEvent>
#include <algorithm>
#include <cmath>

// ==============================================================
// VideoItem
// ==============================================================
VideoItem::VideoItem(QQuickItem* parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true);
    setAcceptedMouseButtons(Qt::LeftButton | Qt::MiddleButton);

    if (VideoEngine* eng = VideoEngine::instance()) {
        connect(eng, &VideoEngine::frameReady, this, &VideoItem::onFrameReady);
    }
}

bool VideoItem::hasVideo() const
{
    VideoEngine* eng = VideoEngine::instance();
    return eng && eng->hasFrame();
}

QSize VideoItem::sourceSize() const
{
    VideoEngine* eng = VideoEngine::instance();
    return eng ? eng->displaySize() : QSize();
}

void VideoItem::setInteractive(bool on)
{
    if (interactive_ == on) return;
    interactive_ = on;
    if (!on && panning) {
        panning = false;
        panButton = Qt::NoButton;
        unsetCursor();
    }
    emit interactiveChanged();
}

void VideoItem::onFrameReady()
{
    const QSize s = sourceSize();
    if (s != lastSize_) {
        lastSize_ = s;
        resetZoom();            // 解像度が変わったら等倍へ
    }
    emit frameChanged();
    update();
}

// 表示枠（アスペクト比を保って中央寄せ）
QRectF VideoItem::videoRect() const
{
    const QSize s = sourceSize();
    if (s.width() <= 0 || s.height() <= 0 || width() <= 0 || height() <= 0)
        return QRectF();

    const double srcAspect = double(s.width()) / s.height();
    const double dstAspect = width() / height();
    if (srcAspect > dstAspect) {
        const double h = width() / srcAspect;
        return QRectF(0, (height() - h) * 0.5, width(), h);
    }
    const double w = height() * srcAspect;
    return QRectF((width() - w) * 0.5, 0, w, height());
}

QRectF VideoItem::viewRect() const
{
    const QSize s = sourceSize();
    if (s.width() <= 0 || s.height() <= 0) return QRectF(0, 0, 1, 1);
    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    return QRectF(sx / s.width(), sy / s.height(), sw / s.width(), sh / s.height());
}

double VideoItem::zoomMax() const
{
    const QSize s = sourceSize();
    if (s.width() <= 0 || s.height() <= 0) return 1.0;
    // 最大倍率: 1 画素が画面上で 32px になるまで（スポイトで画素を狙えるように）
    const QRectF vr = videoRect();
    const double fit = vr.width() > 0 ? vr.width() / s.width() : 1.0;   // 等倍表示での 1 画素の大きさ
    const double byPixel = 32.0 / std::max(1e-6, fit);
    const double legacy  = std::min(s.width() / 320.0, s.height() / 240.0);
    return std::max(1.0, std::max(byPixel, legacy));
}

void VideoItem::resetZoom()
{
    const QSize s = sourceSize();
    zoomFactor = 1.0;
    viewCx = s.width()  * 0.5;
    viewCy = s.height() * 0.5;
    panning = false;
    panButton = Qt::NoButton;
    unsetCursor();
    emit viewChanged();
    update();
}

void VideoItem::clampView()
{
    const QSize s = sourceSize();
    zoomFactor = std::clamp(zoomFactor, 1.0, zoomMax());
    const double sw = s.width()  / zoomFactor;
    const double sh = s.height() / zoomFactor;
    viewCx = std::clamp(viewCx, sw * 0.5, s.width()  - sw * 0.5);
    viewCy = std::clamp(viewCy, sh * 0.5, s.height() - sh * 0.5);
}

void VideoItem::visibleSrcRect(double& sx, double& sy, double& sw, double& sh) const
{
    const QSize s = sourceSize();
    if (zoomFactor <= 1.0 || s.width() <= 0 || s.height() <= 0) {
        sx = 0; sy = 0; sw = s.width(); sh = s.height();
        return;
    }
    sw = s.width()  / zoomFactor;
    sh = s.height() / zoomFactor;
    sx = viewCx - sw * 0.5;
    sy = viewCy - sh * 0.5;
}

QPointF VideoItem::itemToDisplayUV(const QPointF& pos) const
{
    const QRectF r = videoRect();
    if (r.width() <= 0 || r.height() <= 0) return QPointF(0.5, 0.5);
    return QPointF(std::clamp((pos.x() - r.left()) / r.width(),  0.0, 1.0),
                   std::clamp((pos.y() - r.top())  / r.height(), 0.0, 1.0));
}

void VideoItem::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry)
{
    QQuickItem::geometryChange(newGeometry, oldGeometry);
    emit viewChanged();
    update();
}

void VideoItem::wheelEvent(QWheelEvent* event)
{
    if (!interactive_ || !hasVideo()) { event->ignore(); return; }

    const int delta = event->angleDelta().y();
    if (delta == 0) { event->accept(); return; }

    // カーソル下の画像位置を固定して拡大縮小
    const QPointF uv = itemToDisplayUV(event->position());
    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    const double imgX = sx + uv.x() * sw;
    const double imgY = sy + uv.y() * sh;

    const QSize s = sourceSize();
    zoomFactor = std::clamp(zoomFactor * std::pow(2.0, delta / 360.0), 1.0, zoomMax()); // 3ノッチで2倍
    const double nsw = s.width()  / zoomFactor;
    const double nsh = s.height() / zoomFactor;
    viewCx = imgX + (0.5 - uv.x()) * nsw;
    viewCy = imgY + (0.5 - uv.y()) * nsh;
    clampView();

    event->accept();
    emit viewChanged();
    update();
}

void VideoItem::mousePressEvent(QMouseEvent* event)
{
    const bool panBtn = (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton);
    if (interactive_ && panBtn && !panning && zoomFactor > 1.0001) {
        panning = true;
        panButton = event->button();
        panLastPos = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    // ダブルクリックを受けるため左ボタンも受け取る
    event->setAccepted(event->button() == Qt::LeftButton);
}

void VideoItem::mouseMoveEvent(QMouseEvent* event)
{
    if (!panning) { event->ignore(); return; }

    const QRectF r = videoRect();
    if (r.width() > 0 && r.height() > 0) {
        double sx, sy, sw, sh;
        visibleSrcRect(sx, sy, sw, sh);
        const QPointF d = event->position() - panLastPos;
        viewCx -= d.x() * (sw / r.width());
        viewCy -= d.y() * (sh / r.height());
        clampView();
        emit viewChanged();
        update();
    }
    panLastPos = event->position();
    event->accept();
}

void VideoItem::mouseReleaseEvent(QMouseEvent* event)
{
    if (panning && event->button() == panButton) {
        panning = false;
        panButton = Qt::NoButton;
        unsetCursor();
        event->accept();
        return;
    }
    event->ignore();
}

void VideoItem::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && zoomFactor > 1.0001) {
        resetZoom();
        event->accept();
        return;
    }
    event->ignore();
}

QSGNode* VideoItem::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*)
{
    VideoEngine* eng = VideoEngine::instance();
    const QRectF dst = videoRect();
    if (!eng || !eng->hasFrame() || !window() || dst.isEmpty() || eng->displayTexture() == 0) {
        delete oldNode;
        texId_ = 0;
        return nullptr;
    }

    // engine が FBO を描き終えるのを GPU 側で待つ（CPU は止めない）
    eng->waitFrameFence();

    auto* node = static_cast<QSGImageNode*>(oldNode);
    if (!node) {
        node = window()->createImageNode();
        node->setOwnsTexture(true);
        node->setFiltering(QSGTexture::Linear);
        // FBO は下の行から並んでいるので上下反転して貼る
        node->setTextureCoordinatesTransform(QSGImageNode::MirrorVertically);
        texId_ = 0;
    }

    const GLuint id = eng->displayTexture();
    const QSize  sz = eng->displaySize();
    if (id != texId_ || sz != texSize_) {
        // 解像度変更で engine がテクスチャを作り直したら包み直す
        QSGTexture* t = QNativeInterface::QSGOpenGLTexture::fromNative(id, window(), sz);
        node->setTexture(t);
        texId_ = id;
        texSize_ = sz;
    }

    // 表示範囲（元画像は上原点、テクスチャは下原点なので y を読み替える）
    double sx, sy, sw, sh;
    visibleSrcRect(sx, sy, sw, sh);
    node->setSourceRect(QRectF(sx, sz.height() - (sy + sh), sw, sh));
    node->setRect(dst);
    // 拡大表示（1 画素 > 画面 1px）は Nearest でぼかさない。
    //   縮小表示（16K を窓に収めるときなど）は Linear（Nearest だとちらつく）
    const double pxPerSrc = (sw > 0 ? dst.width() / sw : 1.0) * (window() ? window()->effectiveDevicePixelRatio() : 1.0);
    node->setFiltering(pxPerSrc >= 1.5 ? QSGTexture::Nearest : QSGTexture::Linear);
    node->markDirty(QSGNode::DirtyMaterial);
    return node;
}

// ==============================================================
// HistogramItem
// ==============================================================
HistogramItem::HistogramItem(QQuickItem* parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true);
    if (VideoEngine* eng = VideoEngine::instance()) {
        connect(eng, &VideoEngine::histogramChanged, this, [this] {
            emit changed();
            update();
        });
    }
}

// 自分の実ピクセルサイズを engine に伝える（プロットをその解像度で描かせる）
void HistogramItem::pushViewport()
{
    VideoEngine* eng = VideoEngine::instance();
    if (!eng || width() <= 0 || height() <= 0) return;
    const qreal dpr = window() ? window()->effectiveDevicePixelRatio() : 1.0;
    eng->setHistViewport(QSize(qRound(width() * dpr), qRound(height() * dpr)), dpr);
}

void HistogramItem::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry)
{
    QQuickItem::geometryChange(newGeometry, oldGeometry);
    pushViewport();
}

void HistogramItem::itemChange(ItemChange change, const ItemChangeData& value)
{
    QQuickItem::itemChange(change, value);
    if (change == ItemSceneChange || change == ItemDevicePixelRatioHasChanged)
        pushViewport();
}

bool HistogramItem::valid() const
{
    VideoEngine* eng = VideoEngine::instance();
    return eng && eng->histValid();
}

QSGNode* HistogramItem::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*)
{
    VideoEngine* eng = VideoEngine::instance();
    if (!eng || !eng->histValid() || !window() || width() <= 0 || height() <= 0 || eng->histTexture() == 0) {
        delete oldNode;
        texId_ = 0;
        return nullptr;
    }

    eng->waitFrameFence();

    auto* node = static_cast<QSGImageNode*>(oldNode);
    if (!node) {
        node = window()->createImageNode();
        node->setOwnsTexture(true);
        node->setFiltering(QSGTexture::Linear);
        node->setTextureCoordinatesTransform(QSGImageNode::MirrorVertically);
        texId_ = 0;
    }
    // サイズが変わると engine がテクスチャを作り直す（同じ番号が再利用されうるので寸法も見る）
    if (eng->histTexture() != texId_ || eng->histSize() != texSize_) {
        QSGTexture* t = QNativeInterface::QSGOpenGLTexture::fromNative(
            eng->histTexture(), window(), eng->histSize(), QQuickWindow::TextureHasAlphaChannel);
        node->setTexture(t);
        texId_ = eng->histTexture();
        texSize_ = eng->histSize();
    }
    node->setRect(boundingRect());
    node->markDirty(QSGNode::DirtyMaterial);
    return node;
}
