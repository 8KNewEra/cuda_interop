import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// トーンカーブ（RGB 全体 / R / G / B）
//   ・クリックで点を追加、ドラッグで移動、ダブルクリック / 右クリックで削除（両端は消せない）
//   ・両端の点は上下だけ動かせる。点は最大 16 個
//   ・点の間は単調 3 次補間（C++ の ToneCurve::lut と同じ曲線）
ColumnLayout {
    id: root
    property int ch: 0                                        // 0 RGB / 1 R / 2 G / 3 B
    readonly property var names: ["RGB", "R", "G", "B"]
    readonly property var colors: ["#E8EAED", "#FF5A52", "#46D369", "#4A9DFF"]
    readonly property var pts: (player.curves && player.curves[ch]) ? player.curves[ch] : [{x: 0, y: 0}, {x: 255, y: 255}]
    property int sel: -1                                      // 選択中の点（表示用）

    function isIdentity(list) {
        for (const p of list) if (p.x !== p.y) return false
        return true
    }
    function changedCh(i) { return player.curves && player.curves[i] ? !isIdentity(player.curves[i]) : false }

    spacing: 8

    // ---- チャンネルのタブ（変更があれば右上に点）----
    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 30
        radius: 7
        color: Theme.panel2
        border.color: Theme.border
        opacity: enabled ? 1.0 : 0.45
        Row {
            anchors.fill: parent
            anchors.margins: 2
            Repeater {
                model: 4
                delegate: AbstractButton {
                    id: tab
                    required property int index
                    readonly property bool on: root.ch === index
                    width: parent.width / 4
                    height: parent.height
                    hoverEnabled: true
                    focusPolicy: Qt.NoFocus
                    onClicked: { root.ch = index; root.sel = -1 }
                    background: Rectangle { radius: 5; color: tab.on ? Theme.selected : (tab.hovered ? Theme.hover : "transparent") }
                    contentItem: Item {
                        Row {
                            anchors.centerIn: parent
                            spacing: 5
                            Rectangle { visible: tab.index > 0; width: 8; height: 8; radius: 4; color: root.colors[tab.index]; anchors.verticalCenter: parent.verticalCenter }
                            Text {
                                text: root.names[tab.index]
                                color: tab.on ? "#FFFFFF" : Theme.muted2
                                font.family: Theme.mono; font.pixelSize: 12; font.bold: tab.on
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        Rectangle {
                            visible: root.changedCh(tab.index)
                            anchors.right: parent.right; anchors.top: parent.top
                            anchors.rightMargin: 5; anchors.topMargin: 5
                            width: 5; height: 5; radius: 2.5
                            color: Theme.marker
                        }
                    }
                }
            }
        }
    }

    // ---- カーブ編集欄（正方形）----
    Rectangle {
        id: box
        Layout.fillWidth: true
        implicitHeight: width
        radius: 6
        color: "#0B0C0E"
        border.color: Theme.border
        opacity: enabled ? 1.0 : 0.45

        readonly property real pad: 8
        readonly property real side: width - pad * 2
        function toX(v) { return pad + v / 255 * side }
        function toY(v) { return pad + (1 - v / 255) * side }
        function fromX(px) { return Math.round(Math.max(0, Math.min(255, (px - pad) / side * 255))) }
        function fromY(py) { return Math.round(Math.max(0, Math.min(255, (1 - (py - pad) / side) * 255))) }

        Canvas {
            id: canvas
            anchors.fill: parent
            // 曲線は C++ の LUT をそのまま描く（見た目と実際の効き方を一致させる）
            property var lut: player.curveLut(root.ch)
            Connections {
                target: player
                function onFilterChanged() { canvas.lut = player.curveLut(root.ch); canvas.requestPaint() }
            }
            Connections {
                target: root
                function onChChanged() { canvas.lut = player.curveLut(root.ch); canvas.requestPaint() }
                function onSelChanged() { canvas.requestPaint() }
            }
            onWidthChanged: requestPaint()
            onPaint: {
                const g = getContext("2d")
                g.reset()
                const p = box.pad, s = box.side
                // 格子（4 分割）
                g.strokeStyle = "#22262C"; g.lineWidth = 1
                for (let i = 0; i <= 4; i++) {
                    const t = p + s * i / 4
                    g.beginPath(); g.moveTo(p, t); g.lineTo(p + s, t); g.stroke()
                    g.beginPath(); g.moveTo(t, p); g.lineTo(t, p + s); g.stroke()
                }
                // 対角線（変更なしの基準）
                g.strokeStyle = "#3A3F47"; g.setLineDash([4, 4])
                g.beginPath(); g.moveTo(p, p + s); g.lineTo(p + s, p); g.stroke()
                g.setLineDash([])
                // 曲線
                if (lut && lut.length === 256) {
                    g.strokeStyle = root.colors[root.ch]; g.lineWidth = 2
                    g.beginPath()
                    for (let v = 0; v < 256; v++) {
                        const x = box.toX(v), y = box.toY(lut[v])
                        if (v === 0) g.moveTo(x, y); else g.lineTo(x, y)
                    }
                    g.stroke()
                }
                // 点
                for (let i = 0; i < root.pts.length; i++) {
                    const q = root.pts[i]
                    g.beginPath()
                    g.arc(box.toX(q.x), box.toY(q.y), i === root.sel ? 6 : 4.5, 0, Math.PI * 2)
                    g.fillStyle = i === root.sel ? "#4FD1C5" : "#FFFFFF"
                    g.fill()
                    g.strokeStyle = "#0B0C0E"; g.lineWidth = 1.5; g.stroke()
                }
            }
        }

        MouseArea {
            id: ma
            anchors.fill: parent
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            hoverEnabled: true
            preventStealing: true
            property int drag: -1                 // ドラッグ中の点（root.pts の添字）
            property var work: []                 // ドラッグ中の点の写し

            function hit(mx, my) {
                for (let i = 0; i < root.pts.length; i++) {
                    const q = root.pts[i]
                    if (Math.abs(box.toX(q.x) - mx) <= 9 && Math.abs(box.toY(q.y) - my) <= 9) return i
                }
                return -1
            }
            function copyPts() { return root.pts.map(q => ({ x: q.x, y: q.y })) }
            function removeAt(i) {
                if (i <= 0 || i >= root.pts.length - 1) return      // 両端は消さない
                const a = copyPts(); a.splice(i, 1)
                root.sel = -1
                player.setCurvePoints(root.ch, a)
            }

            cursorShape: drag >= 0 ? Qt.ClosedHandCursor : (hit(mouseX, mouseY) >= 0 ? Qt.OpenHandCursor : Qt.CrossCursor)

            onPressed: (m) => {
                if (m.button === Qt.RightButton) { removeAt(hit(m.x, m.y)); return }
                let i = hit(m.x, m.y)
                work = copyPts()
                if (i < 0) {
                    // 新しい点（最大 16 個。両端の x には置かない）
                    if (work.length >= 16) return
                    const x = box.fromX(m.x)
                    if (x <= 0 || x >= 255 || work.some(q => q.x === x)) return
                    work.push({ x: x, y: box.fromY(m.y) })
                    work.sort((a, b) => a.x - b.x)
                    i = work.findIndex(q => q.x === x)
                    player.setCurvePoints(root.ch, work)
                }
                drag = i
                root.sel = i
            }
            onPositionChanged: (m) => {
                if (drag < 0 || !pressed) return
                const last = work.length - 1
                let x = box.fromX(m.x), y = box.fromY(m.y)
                // 両端は上下だけ。途中の点は隣の点を越えない
                if (drag === 0) x = 0
                else if (drag === last) x = 255
                else x = Math.max(work[drag - 1].x + 1, Math.min(work[drag + 1].x - 1, x))
                work[drag] = { x: x, y: y }
                player.setCurvePoints(root.ch, work)
            }
            onReleased: drag = -1
            onDoubleClicked: (m) => removeAt(hit(m.x, m.y))
        }
    }

    // ---- 選択中の点の値 ／ リセット ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            Layout.fillWidth: true
            text: root.sel >= 0 && root.sel < root.pts.length
                  ? "入力 " + root.pts[root.sel].x + "  →  出力 " + root.pts[root.sel].y
                  : "制御点 " + root.pts.length + " / 16"
            color: enabled ? Theme.text2 : Theme.faint
            font.family: Theme.mono; font.pixelSize: 11
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Repeater {
            model: [{ text: "このカーブをリセット", all: false }, { text: "すべてリセット", all: true }]
            delegate: AbstractButton {
                id: rb
                required property var modelData
                Layout.fillWidth: true
                implicitHeight: 28
                text: modelData.text
                enabled: modelData.all ? [0, 1, 2, 3].some(i => root.changedCh(i)) : root.changedCh(root.ch)
                hoverEnabled: true
                focusPolicy: Qt.NoFocus
                opacity: enabled ? 1.0 : 0.4
                onClicked: { root.sel = -1; modelData.all ? player.resetAllCurves() : player.resetCurve(root.ch) }
                background: Rectangle { radius: 7; color: rb.hovered && rb.enabled ? Theme.hover : Theme.panel2; border.color: Theme.border }
                contentItem: Text {
                    text: rb.text
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    color: Theme.text2; font.family: Theme.font; font.pixelSize: 12
                }
            }
        }
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: "クリックで制御点を追加、ドラッグで移動、ダブルクリックまたは右クリックで削除します。RGB カーブを適用した後に、各チャンネルのカーブを適用します。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
