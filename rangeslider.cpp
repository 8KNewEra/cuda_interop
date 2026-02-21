#include "RangeSlider.h"
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>

RangeSlider::RangeSlider(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(30);
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

    update();

    emit rangeStartChanged(m_start);
    emit rangeEndChanged(m_end);
}

int RangeSlider::pixelPosFromValue(int value) const
{
    if (m_max == m_min) return 0;

    double ratio = double(value - m_min) / (m_max - m_min);
    return int(ratio * width());
}

int RangeSlider::valueFromPixelPos(int pos) const
{
    if (width() == 0) return m_min;

    double ratio = double(pos) / width();
    return m_min + int(ratio * (m_max - m_min));
}

QRect RangeSlider::handleRect(int value) const
{
    int x = pixelPosFromValue(value);
    int handleWidth = 14;
    return QRect(x - handleWidth/2, height()/2 - 10, handleWidth, 20);
}

void RangeSlider::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int h = height();

    int startPos = pixelPosFromValue(m_start);
    int endPos   = pixelPosFromValue(m_end);
    int playPos  = pixelPosFromValue(m_play);

    // 背景バー
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(120,120,120));
    p.drawRect(0, h/2 - 3, width(), 6);

    // 範囲
    p.setBrush(QColor(50,150,255));
    p.drawRect(startPos, h/2 - 4, endPos - startPos, 8);

    // 再生位置ライン
    p.setPen(QPen(Qt::red, 2));
    p.drawLine(playPos, 0, playPos, height());

    // ハンドル
    p.setPen(Qt::NoPen);

    p.setBrush(Qt::white);
    p.drawEllipse(handleRect(m_start));

    p.setBrush(Qt::white);
    p.drawEllipse(handleRect(m_end));

    // playハンドル
    p.setBrush(Qt::red);
    p.drawEllipse(handleRect(m_play));
}

void RangeSlider::mousePressEvent(QMouseEvent* event)
{
    // play優先
    if (handleRect(m_play).contains(event->pos()))
        m_activeHandle = PlayHandle;
    else if (handleRect(m_start).contains(event->pos()))
        m_activeHandle = StartHandle;
    else if (handleRect(m_end).contains(event->pos()))
        m_activeHandle = EndHandle;
    else
        m_activeHandle = NoHandle;

    if (m_activeHandle == PlayHandle)
        m_userInteraction = true;
}

void RangeSlider::mouseMoveEvent(QMouseEvent* event)
{
    if (m_activeHandle == NoHandle)
        return;

    int value = valueFromPixelPos(event->pos().x());

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

void RangeSlider::mouseReleaseEvent(QMouseEvent*)
{
    m_userInteraction = false;
    m_activeHandle = NoHandle;
}
