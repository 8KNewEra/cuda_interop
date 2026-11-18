import QtQuick
import QtQuick.Controls

// ON/OFF スイッチ（40x24）
AbstractButton {
    id: root
    checkable: true
    implicitWidth: 40
    implicitHeight: 24
    focusPolicy: Qt.NoFocus
    hoverEnabled: true
    opacity: enabled ? 1.0 : 0.4

    background: Rectangle {
        radius: 12
        color: root.checked ? Theme.accent : Theme.dashed
        Behavior on color { ColorAnimation { duration: 120 } }
    }
    contentItem: Item {
        Rectangle {
            width: 18; height: 18; radius: 9
            y: 3
            x: root.checked ? 19 : 3
            color: "#FFFFFF"
            Behavior on x { NumberAnimation { duration: 120 } }
        }
    }
}
