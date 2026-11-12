import QtQuick
import QtQuick.Controls

// 小さなセグメント切り替え [ A | B | C ]
//   options = [{text, value}]、value = 選択中。押すと picked(v)
Rectangle {
    id: root
    property var options: []
    property var value
    property bool mono: false
    signal picked(var v)

    implicitHeight: 30
    radius: 7
    color: Theme.panel2
    border.color: Theme.border
    opacity: enabled ? 1.0 : 0.45

    Row {
        anchors.fill: parent
        anchors.margins: 2
        Repeater {
            model: root.options
            delegate: AbstractButton {
                id: seg
                required property var modelData
                readonly property bool on: root.value == modelData.value
                width: parent.width / Math.max(1, root.options.length)
                height: parent.height
                hoverEnabled: true
                focusPolicy: Qt.NoFocus
                onClicked: root.picked(modelData.value)
                background: Rectangle { radius: 5; color: seg.on ? Theme.selected : (seg.hovered ? Theme.hover : "transparent") }
                contentItem: Text {
                    text: seg.modelData.text
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                    color: seg.on ? "#FFFFFF" : Theme.muted2
                    font.family: root.mono ? Theme.mono : Theme.font
                    font.pixelSize: root.mono ? 11 : 12
                }
            }
        }
    }
}
