import QtQuick
import QtQuick.Controls

// アプリ共通のドロップダウンメニュー（パネルと同じ色・角丸・文字）
//   項目は Action で並べる（delegate を効かせるため）。区切りは AppMenuSeparator
Menu {
    id: menu
    padding: 6
    implicitWidth: 220
    topPadding: 6; bottomPadding: 6

    enter: Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 90 } }
    exit:  Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 70 } }

    background: Rectangle {
        implicitWidth: 220
        color: Theme.panel2
        border.color: Theme.border
        radius: 10
    }

    delegate: MenuItem {
        id: mi
        implicitHeight: 32
        leftPadding: 12; rightPadding: 12
        hoverEnabled: true
        background: Rectangle {
            radius: 6
            color: mi.highlighted && mi.enabled ? Theme.hover : "transparent"
        }
        contentItem: Item {
            implicitWidth: lbl.implicitWidth + sc.implicitWidth + 24
            Text {
                id: lbl
                anchors.left: parent.left; anchors.verticalCenter: parent.verticalCenter
                text: mi.text
                font.family: Theme.font; font.pixelSize: 13
                color: mi.enabled ? Theme.text2 : Theme.faint
            }
            Text {
                id: sc
                anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter
                text: mi.action && mi.action.hint ? mi.action.hint : ""
                font.family: Theme.mono; font.pixelSize: 11
                color: Theme.faint
            }
        }
        // 既定の矢印・チェック表示は使わない
        indicator: null
        arrow: null
    }
}
