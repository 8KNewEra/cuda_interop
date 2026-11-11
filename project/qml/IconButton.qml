import QtQuick
import QtQuick.Controls

// 44x44 のアイコンボタン（トランスポート用）。caption があれば下に小さく出す。
AbstractButton {
    id: root
    property var    paths: []
    property color  iconColor: Theme.text2
    property real   iconSize: 18
    property real   strokeWidth: 2
    property bool   filled: false
    property string caption: ""
    property string tip: ""

    implicitWidth: 44
    implicitHeight: 44
    hoverEnabled: true
    focusPolicy: Qt.NoFocus
    opacity: enabled ? 1.0 : 0.35

    ToolTip.visible: hovered && tip.length > 0
    ToolTip.delay: 500
    ToolTip.text: tip

    background: Rectangle {
        radius: 10
        color: root.down ? Theme.selected : (root.hovered ? Theme.hover : "transparent")
    }

    contentItem: Item {
        Icon {
            anchors.centerIn: parent
            anchors.verticalCenterOffset: root.caption.length > 0 ? -4 : 0
            paths: root.paths
            size: root.iconSize
            strokeWidth: root.strokeWidth
            filled: root.filled
            color: root.hovered ? "#FFFFFF" : root.iconColor
        }
        Text {
            visible: root.caption.length > 0
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 2
            text: root.caption
            color: Theme.muted
            font.family: Theme.mono
            font.pixelSize: 10
        }
    }
}
