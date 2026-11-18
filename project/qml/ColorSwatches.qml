import QtQuick
import QtQuick.Controls

// 色見本の並び（1 つ選ぶ）。colors = [{text, value, c}]、value = 選択中。押すと picked(v)
Row {
    id: root
    property var colors: []
    property var value
    signal picked(var v)
    spacing: 8
    opacity: enabled ? 1.0 : 0.4

    Repeater {
        model: root.colors
        delegate: AbstractButton {
            id: sw
            required property var modelData
            readonly property bool on: root.value === modelData.value
            width: 26; height: 26
            hoverEnabled: true
            focusPolicy: Qt.NoFocus
            onClicked: root.picked(modelData.value)
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
