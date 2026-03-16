#include "RangeSlider.h"
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>

RangeSlider::RangeSlider(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(30);

    //マウス追跡
    setMouseTracking(true);
}

void RangeSlider::setRange(int min, int max)
{
    if (min >= max) return;

    m_min = min;
    m_max = max;

    m_start = qBound(m_min, m_start, m_max);
    m_end   = qBound(m_min, m_end, m_max);
    m_play  = qBound(m_min, m_play, m_max);

    update();
}

void RangeSlider::setMinimum(int min)
{
    setRange(min, m_max);
}

void RangeSlider::setMaximum(int max)
{
    setRange(m_min, max);
}

void RangeSlider::setStartValue(int value)
{
    value = qBound(m_min, value, m_end - m_minSpacing);

    if (m_start == value)
        return;

    m_start = value;

    bool playChanged = false;

    if (m_play < m_start)
    {
        m_play = m_start;
        playChanged = true;

        //一旦マウス追跡解除
        setMouseTracking(false);
    }

    update();

    emit rangeStartChanged(m_start);

    if (playChanged)
        emit playValueChanged(m_play);
}

void RangeSlider::setEndValue(int value)
{
    value = qBound(m_start + m_minSpacing, value, m_max);

    if (m_end == value)
        return;

    m_end = value;

    bool playChanged = false;

    if (m_play > m_end)
    {
        m_play = m_end;
        playChanged = true;

        //一旦マウス追跡解除
        setMouseTracking(false);
    }

    update();

    emit rangeEndChanged(m_end);

    if (playChanged)
        emit playValueChanged(m_play);
}

void RangeSlider::setPlayValue(int value)
{
    // ★ ユーザー操作中は外部更新を拒否
    if (!m_userInteraction)
    {
        value = qBound(m_start, value, m_end);

        if (m_play == value)
            return;

        m_play = value;
        update();
    }
}

void RangeSlider::setValues(int start, int end)
{
    if (start > end) return;

    m_start = qBound(m_min, start, m_max);
    m_end   = qBound(m_min, end, m_max);
    m_play = qBound(m_start, m_play, m_end);  // ★追加

    update();

    emit rangeStartChanged(m_start);
    emit rangeEndChanged(m_end);
}

int RangeSlider::pixelPosFromValue(int value) const
{
    if (m_max == m_min)
        return handleWidth;

    int sliderMin = handleWidth;
    int sliderMax = width() - handleWidth;

    double ratio = double(value - m_min) / (m_max - m_min);

    return sliderMin + int(ratio * (sliderMax - sliderMin));
}

int RangeSlider::valueFromPixelPos(int pos) const
{
    int sliderMin = handleWidth;
    int sliderMax = width() - handleWidth;

    pos = qBound(sliderMin, pos, sliderMax);

    double ratio = double(pos - sliderMin) / (sliderMax - sliderMin);

    return m_min + int(ratio * (m_max - m_min));
}

QRect RangeSlider::handleRectPlay(int value) const
{
    int x = pixelPosFromValue(value);

    int hitWidth = 12;   // 当たり判定幅
    int hitHeight = height();

    return QRect(x - hitWidth/2, 0, hitWidth, hitHeight);
}

QRect RangeSlider::handleRectStart(int value) const
{
    int x = pixelPosFromValue(value);
    return QRect(x - handleWidth, height()/2 - handleHeight/2,
                 handleWidth, handleHeight);
}

QRect RangeSlider::handleRectEnd(int value) const
{
    int x = pixelPosFromValue(value);
    return QRect(x, height()/2 - handleHeight/2,
                 handleWidth, handleHeight);
}

void RangeSlider::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int h = height();

    int startPos = pixelPosFromValue(m_start);
    int endPos   = pixelPosFromValue(m_end);
    int playPos  = pixelPosFromValue(m_play);

    int sliderY = h/2 - 3;

    // 背景バー
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255,255,255,40));
    p.drawRoundedRect(handleWidth, sliderY,
                      width()-handleWidth*2, 6, 3,3);

    // 範囲
    p.setBrush(QColor(90,170,255));
    p.drawRoundedRect(startPos, sliderY,
                      endPos - startPos, 6, 3,3);

    // start ハンドル
    p.setBrush(Qt::white);
    p.drawRoundedRect(handleRectStart(m_start),6,6);

    // end ハンドル
    p.drawRoundedRect(handleRectEnd(m_end),6,6);

    // play ヘッド（棒）
    QRect hit = handleRectPlay(m_play);

    int playWidth = 5;  // 見た目の幅

    QRect drawRect(
        hit.center().x() - playWidth/2,
        0,
        playWidth,
        height()
        );

    p.setBrush(Qt::red);
    p.setPen(Qt::NoPen);
    p.drawRoundedRect(drawRect,2,2);
}

void RangeSlider::mousePressEvent(QMouseEvent* event)
{
    int value = valueFromPixelPos(event->pos().x());

    // play優先
    if (handleRectPlay(m_play).contains(event->pos()))
        m_activeHandle = PlayHandle;

    else if (handleRectStart(m_start).contains(event->pos()))
        m_activeHandle = StartHandle;

    else if (handleRectEnd(m_end).contains(event->pos()))
        m_activeHandle = EndHandle;

    // ★スライダーバークリック → シーク
    else
    {
        if (value >= m_start && value <= m_end)
        {
            m_play = value;
            emit playValueChanged(m_play);
            emit playValueReleaseChanged(m_play);
            update();
        }

        m_activeHandle = NoHandle;
        return;
    }

    // ドラッグカーソル
    if (m_activeHandle != NoHandle)
        setCursor(Qt::PointingHandCursor);

    if (m_activeHandle == PlayHandle)
        m_userInteraction = true;
}

void RangeSlider::mouseMoveEvent(QMouseEvent* event)
{
    //マウス追跡開始
    setMouseTracking(true);

    int value = valueFromPixelPos(event->pos().x());

    // ★ドラッグ中
    if (m_activeHandle != NoHandle)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else
    {
        // ★ハンドル or 範囲バーなら指カーソル
        if (handleRectPlay(m_play).contains(event->pos()) ||
            handleRectStart(m_start).contains(event->pos()) ||
            handleRectEnd(m_end).contains(event->pos()) ||
            (value >= m_start && value <= m_end))
        {
            setCursor(Qt::PointingHandCursor);
        }
        else
        {
            setCursor(Qt::ArrowCursor);
        }
    }

    if (m_activeHandle == NoHandle)
        return;

    switch (m_activeHandle)
    {
    case StartHandle:
        setStartValue(value);
        break;

    case PlayHandle:
        value = qBound(m_start, value, m_end);
        if (m_play != value)
        {
            m_play = value;
            emit playValueChanged(m_play);
        }
        break;

    case EndHandle:
        setEndValue(value);
        break;

    default:
        break;
    }

    update();
}

void RangeSlider::leaveEvent(QEvent *)
{
    if (m_activeHandle == NoHandle)   // ドラッグ中は戻さない
        setCursor(Qt::ArrowCursor);
}

void RangeSlider::mouseReleaseEvent(QMouseEvent*)
{
    if(m_activeHandle == PlayHandle)emit playValueReleaseChanged(m_play);
    m_userInteraction = false;
    m_activeHandle = NoHandle;
}
