import QtQuick

// 推移グラフ（右端が最新。動画情報・エンコード設定の使用率表示で共用）
//   記録は呼び出し側が持ち、1 秒ごとに新しい配列を渡す（capacity 60 = 直近 60 秒）
//   values: 古い → 新しい順の配列。null は欠け（線を切る）
//   maxValue: 縦軸の上限、refValue >= 0 なら基準線（破線）を引く
Canvas {
    id: plot
    property int   capacity: 60          // 横軸の点数（右端が最新）
    property var   values: []
    property real  maxValue: 100
    property real  refValue: -1
    property color lineColor: Theme.accent
    onValuesChanged: requestPaint()
    onMaxValueChanged: requestPaint()
    onRefValueChanged: requestPaint()
    onLineColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onPaint: {
        const ctx = getContext("2d")
        ctx.reset()
        const w = width, h = height
        ctx.fillStyle = Theme.border
        ctx.globalAlpha = 0.45
        ctx.fillRect(0, 0, w, h)
        ctx.globalAlpha = 1
        const mx = maxValue > 0 ? maxValue : 1
        const yOf = (v) => h - 1 - Math.max(0, Math.min(1, v / mx)) * (h - 2)
        if (refValue >= 0) {
            ctx.strokeStyle = Theme.muted2
            ctx.lineWidth = 1
            ctx.setLineDash([3, 3])
            ctx.beginPath(); ctx.moveTo(0, yOf(refValue)); ctx.lineTo(w, yOf(refValue)); ctx.stroke()
            ctx.setLineDash([])
        }
        const n = values.length
        if (n === 0) return
        const step = w / Math.max(1, capacity - 1)
        const x0 = w - (n - 1) * step
        // 塗り（区間ごと）と線
        let seg = []
        const flush = () => {
            if (seg.length === 0) return
            ctx.globalAlpha = 0.18
            ctx.fillStyle = lineColor
            ctx.beginPath()
            ctx.moveTo(seg[0][0], h)
            for (const p of seg) ctx.lineTo(p[0], p[1])
            ctx.lineTo(seg[seg.length - 1][0], h)
            ctx.closePath(); ctx.fill()
            ctx.globalAlpha = 1
            ctx.strokeStyle = lineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            ctx.moveTo(seg[0][0], seg[0][1])
            for (const p of seg) ctx.lineTo(p[0], p[1])
            if (seg.length === 1) ctx.lineTo(seg[0][0] + 1, seg[0][1])
            ctx.stroke()
            seg = []
        }
        for (let i = 0; i < n; ++i) {
            const v = values[i]
            if (v === null || v === undefined) { flush(); continue }
            seg.push([x0 + i * step, yOf(v)])
        }
        flush()
    }
}
