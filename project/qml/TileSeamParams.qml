import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// タイル境界チェック（境界線の表示 ＋ つなぎ目の段差の強調）
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    readonly property int cols: player.videoInfo.tileCols || 1
    readonly property int rows: player.videoInfo.tileRows || 1
    readonly property bool tiled: cols * rows > 1
    spacing: 10

    Text {
        Layout.fillWidth: true
        text: root.tiled ? "タイル構成  " + root.cols + " × " + root.rows + "（" + (root.cols * root.rows) + " タイル）"
                         : "この動画はタイル分割されていません。"
        color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12
    }

    RowLayout {
        Layout.fillWidth: true
        enabled: root.tiled
        spacing: 8
        Text { Layout.fillWidth: true; text: "境界線を表示"; color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ToggleSwitch { checked: root.p.seamGrid; onToggled: player.setFilterParam("seamGrid", checked) }
    }
    RowLayout {
        Layout.fillWidth: true
        enabled: root.tiled && root.p.seamGrid
        spacing: 8
        Text { text: "色"; Layout.preferredWidth: 40; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ColorSwatches {
            spacing: 6
            colors: Theme.seamColors
            value: root.p.seamGridColor
            onPicked: (v) => player.setFilterParam("seamGridColor", v)
        }
    }

    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSoft }

    RowLayout {
        Layout.fillWidth: true
        enabled: root.tiled
        spacing: 8
        Text { Layout.fillWidth: true; text: "段差を強調"; color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ToggleSwitch { checked: root.p.seamHighlight; onToggled: player.setFilterParam("seamHighlight", checked) }
    }
    RowLayout {
        Layout.fillWidth: true
        enabled: root.tiled && root.p.seamHighlight
        spacing: 8
        Text { text: "色"; Layout.preferredWidth: 40; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ColorSwatches {
            spacing: 6
            colors: Theme.seamColors
            value: root.p.seamHighlightColor
            onPicked: (v) => player.setFilterParam("seamHighlightColor", v)
        }
    }
    MiniSlider {
        Layout.fillWidth: true
        enabled: root.tiled && root.p.seamHighlight
        labelWidth: 64
        fieldWidth: 58
        label: "しきい値"
        from: 1; to: 64; defaultValue: 8
        value: root.p.seamThreshold
        onMoved: (v) => player.setFilterParam("seamThreshold", v)
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: "境界をまたぐ画素の差が、境界の内側の差よりしきい値（8bit 階調）以上大きい箇所を指定色で表示します。"
              + "判定には補正前の映像を使用します。平坦な領域ではずれを検出できない場合があります。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
