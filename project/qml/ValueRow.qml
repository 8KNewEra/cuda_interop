import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ラベル ＋ スライダー ＋ 数値欄（単位つき）の 1 行。
//   値の本流は C++ 側。スライダーを動かすと moved(v) を出し、
//   範囲やクランプを通った結果が value に戻ってくる。
RowLayout {
    id: root
    property string label: ""
    property string tip: ""
    property int    from: 0
    property int    to: 100
    property int    stepSize: 1
    property int    value: 0
    property string unit: ""
    signal moved(int v)

    spacing: 12
    opacity: enabled ? 1.0 : 0.45

    Text {
        Layout.preferredWidth: 132
        text: root.label
        color: Theme.muted
        font.family: Theme.font
        font.pixelSize: 13
        elide: Text.ElideRight
        ToolTip.visible: tipArea.containsMouse && root.tip.length > 0
        ToolTip.delay: 600
        ToolTip.text: root.tip
        MouseArea { id: tipArea; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
    }

    Slider {
        id: slider
        Layout.fillWidth: true
        from: root.from
        to: Math.max(root.from + 1, root.to)
        stepSize: root.stepSize
        snapMode: Slider.SnapAlways
        value: root.value
        onMoved: root.moved(Math.round(value))

        background: Rectangle {
            x: slider.leftPadding
            y: slider.topPadding + slider.availableHeight / 2 - 2
            width: slider.availableWidth
            height: 4
            radius: 2
            color: Theme.border
            Rectangle {
                width: slider.visualPosition * parent.width
                height: 4
                radius: 2
                color: Theme.accent
            }
        }
        handle: Rectangle {
            x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
            y: slider.topPadding + slider.availableHeight / 2 - height / 2
            width: 16; height: 16; radius: 8
            color: "#FFFFFF"
            border.width: slider.pressed ? 4 : 0
            border.color: Qt.rgba(79/255, 209/255, 197/255, 0.35)
        }
    }

    // 数値欄（スピンボックス）
    NumberSpin {
        Layout.preferredWidth: 176
        value: root.value
        from: root.from
        to: root.to
        stepSize: root.stepSize
        unit: root.unit
        onMoved: (v) => root.moved(v)
    }
}
