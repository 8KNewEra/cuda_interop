import QtQuick
import QtQuick.Layouts
import CudaViewer

// ヒストグラム。プロットは VideoEngine が GL で描いたテクスチャ（HistogramItem）、
// 目盛りと統計値はここで描く
Rectangle {
    id: root
    property var stats: ({})

    width: 360
    implicitHeight: col.implicitHeight + 28
    radius: Theme.radius
    color: Theme.overlay
    border.color: Theme.border

    component H: Text { color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.fillWidth: true }
    component N: Text { color: Theme.text; font.family: Theme.mono; font.pixelSize: 12; Layout.fillWidth: true }

    function fmt(v, d) { return (v === undefined || v === null) ? "-" : Number(v).toFixed(d) }

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: 14
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            Text { text: "ヒストグラム"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.bold: true }
            Item { Layout.fillWidth: true }
            Text { text: "Max Y"; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12 }
            Text {
                text: root.stats.maxY !== undefined ? Number(root.stats.maxY).toLocaleString(Qt.locale("en_US"), 'f', 0) : "-"
                color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
            }
        }

        // プロット（左に Y 目盛り、下に X 目盛り）
        Item {
            id: plotArea
            Layout.fillWidth: true
            // 目盛りの文字はパネル幅に合わせて 12〜15px
            readonly property int tickPx: Math.max(12, Math.min(15, Math.round(root.width / 42)))
            readonly property int gutter: tickPx * 5 + 4
            // プロットはパネル幅に合わせて 3:2 で伸ばす
            readonly property real plotH: Math.round((width - gutter) / 1.5)
            Layout.preferredHeight: plotH + tickPx + 8

            Rectangle {
                id: plotBg
                x: plotArea.gutter
                width: parent.width - plotArea.gutter
                height: plotArea.plotH
                radius: 6
                color: Theme.bg
                border.color: Theme.border
            }
            HistogramItem {
                id: plot
                x: plotBg.x + 1; y: plotBg.y + 1
                width: plotBg.width - 2
                height: plotBg.height - 2
            }
            Text {
                anchors.centerIn: plotBg
                visible: !plot.valid
                text: "計算中…"
                color: Theme.faint; font.family: Theme.font; font.pixelSize: 12
            }

            // Y 目盛り（0 とグリッド線の値）
            Repeater {
                model: root.stats.maxY > 0 ? [0].concat(root.stats.labels || []) : []
                delegate: Text {
                    required property var modelData
                    width: plotArea.gutter - 6
                    horizontalAlignment: Text.AlignRight
                    y: plot.y + plot.height * (1 - modelData / root.stats.maxY) - implicitHeight / 2
                    text: Number(modelData).toLocaleString(Qt.locale("en_US"), 'f', 0)
                    color: Theme.muted2; font.family: Theme.mono; font.pixelSize: plotArea.tickPx
                }
            }

            // X 目盛り（グリッド線の位置に合わせる）
            Repeater {
                model: [0, 64, 128, 192, 256]
                delegate: Text {
                    required property int modelData
                    y: plotBg.height + 3
                    x: Math.max(plotBg.x, Math.min(plotBg.x + plotBg.width - implicitWidth,
                         plot.x + plot.width * modelData / 256 - implicitWidth / 2))
                    text: modelData
                    color: Theme.muted2; font.family: Theme.mono; font.pixelSize: plotArea.tickPx
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            Layout.topMargin: 2
            columns: 4
            columnSpacing: 10
            rowSpacing: 4

            Item { Layout.preferredWidth: 16 } H { text: "min" } H { text: "max" } H { text: "avg" }
            Text { text: "R"; color: Theme.histR; font.family: Theme.mono; font.bold: true; font.pixelSize: 12 }
            N { text: root.fmt(root.stats.minR, 0) } N { text: root.fmt(root.stats.maxR, 0) } N { text: root.fmt(root.stats.avgR, 1) }
            Text { text: "G"; color: Theme.histG; font.family: Theme.mono; font.bold: true; font.pixelSize: 12 }
            N { text: root.fmt(root.stats.minG, 0) } N { text: root.fmt(root.stats.maxG, 0) } N { text: root.fmt(root.stats.avgG, 1) }
            Text { text: "B"; color: Theme.histB; font.family: Theme.mono; font.bold: true; font.pixelSize: 12 }
            N { text: root.fmt(root.stats.minB, 0) } N { text: root.fmt(root.stats.maxB, 0) } N { text: root.fmt(root.stats.avgB, 1) }
        }
    }
}
