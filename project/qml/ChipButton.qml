import QtQuick
import QtQuick.Controls

// 角丸のテキスト＋アイコンボタン（速度・音量・画像処理など）
AbstractButton {
    id: root
    property var    paths: []
    property bool   checkedLook: false
    property bool   primary: false
    property bool   mono: false
    property int    badge: -1          // 0 以上なら右に件数バッジ
    property string tip: ""

    implicitHeight: 34
    implicitWidth: row.implicitWidth + 24
    hoverEnabled: true
    focusPolicy: Qt.NoFocus
    opacity: enabled ? 1.0 : 0.4

    ToolTip.visible: hovered && tip.length > 0
    ToolTip.delay: 500
    ToolTip.text: tip

    background: Rectangle {
        radius: 8
        color: root.primary ? (root.hovered ? Qt.lighter(Theme.accent, 1.08) : Theme.accent)
             : root.checkedLook ? Theme.hover
             : (root.hovered ? Theme.hover : Theme.panel2)
        border.width: root.primary ? 0 : 1
        border.color: root.checkedLook ? Theme.dashed : Theme.border
    }

    contentItem: Item {
        Row {
            id: row
            anchors.centerIn: parent
            spacing: 8
            Icon {
                visible: root.paths.length > 0
                anchors.verticalCenter: parent.verticalCenter
                paths: root.paths
                size: 16
                strokeWidth: root.primary ? 2.2 : 2
                color: root.primary ? Theme.accentInk : Theme.text2
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: root.text
                color: root.primary ? Theme.accentInk : Theme.text
                font.family: root.mono ? Theme.mono : Theme.font
                font.pixelSize: 13
                font.weight: root.primary ? Font.Bold : Font.Medium
            }
            Rectangle {
                visible: root.badge >= 0
                anchors.verticalCenter: parent.verticalCenter
                width: Math.max(20, badgeText.implicitWidth + 12)
                height: 20
                radius: 10
                color: root.badge > 0 ? Theme.accent : Theme.border
                Text {
                    id: badgeText
                    anchors.centerIn: parent
                    text: root.badge
                    color: root.badge > 0 ? Theme.accentInk : Theme.muted
                    font.family: Theme.mono
                    font.pixelSize: 11
                }
            }
        }
    }
}
