import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ソーベルフィルタの設定（エッジのみ / ピーキング）
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    readonly property bool peaking: p.sobelMode === 1
    readonly property var colors: [
        { text: "赤", value: 0, c: "#FF3B30" }, { text: "緑", value: 1, c: "#34C759" },
        { text: "青", value: 2, c: "#0A84FF" }, { text: "黄", value: 3, c: "#FFD60A" },
        { text: "白", value: 4, c: "#FFFFFF" }
    ]
    spacing: 10

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text { text: "表示"; Layout.preferredWidth: 64; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Segmented {
            Layout.fillWidth: true
            options: [{ text: "エッジ表示", value: 0 }, { text: "ピーキング", value: 1 }]
            value: root.p.sobelMode
            onPicked: (v) => player.setFilterParam("sobelMode", v)
        }
    }

    MiniSlider {
        visible: root.peaking
        Layout.fillWidth: true
        label: "しきい値"
        from: 0; to: 100
        value: root.p.peakingThreshold
        onMoved: (v) => player.setFilterParam("peakingThreshold", v)
    }

    RowLayout {
        visible: root.peaking
        Layout.fillWidth: true
        spacing: 8
        Text { text: "色"; Layout.preferredWidth: 64; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Row {
            spacing: 8
            opacity: enabled ? 1.0 : 0.4
            Repeater {
                model: root.colors
                delegate: AbstractButton {
                    id: sw
                    required property var modelData
                    readonly property bool on: root.p.peakingColor === modelData.value
                    width: 26; height: 26
                    hoverEnabled: true
                    focusPolicy: Qt.NoFocus
                    onClicked: player.setFilterParam("peakingColor", modelData.value)
                    ToolTip.visible: hovered
                    ToolTip.delay: 400
                    ToolTip.text: modelData.text
                    background: Rectangle {
                        radius: 13
                        color: "transparent"
                        border.width: sw.on ? 2 : 1
                        border.color: sw.on ? Theme.text : (sw.hovered ? Theme.muted : Theme.border)
                        Rectangle { anchors.centerIn: parent; width: 16; height: 16; radius: 8; color: sw.modelData.c }
                    }
                }
            }
        }
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: root.peaking ? "しきい値を超える輪郭を指定色で強調します。ノイズが目立つ場合は、ガウシアンフィルタ（ぼかし）を弱く併用してください。"
                           : "輪郭の強さをモノクロで表示します。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
