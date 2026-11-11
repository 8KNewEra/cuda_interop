import QtQuick

// ズーム位置（旧: drawZoomNavigator）。外枠=全体、内枠=表示範囲
Rectangle {
    id: root
    property real   zoom: 1.0
    property rect   view: Qt.rect(0, 0, 1, 1)   // 0..1
    property size   source: Qt.size(16, 9)

    readonly property real aspect: source.height > 0 ? source.width / source.height : 16 / 9

    width: 168
    height: map.height + 20 + 22
    radius: 10
    color: Theme.overlay
    border.color: Theme.border

    Rectangle {
        id: map
        x: 10; y: 10
        width: parent.width - 20
        height: width / root.aspect
        radius: 4
        color: Theme.panel2
        border.color: Theme.border

        Rectangle {
            x: map.width * root.view.x
            y: map.height * root.view.y
            width: Math.max(4, map.width * root.view.width)
            height: Math.max(4, map.height * root.view.height)
            color: "transparent"
            border.color: Theme.accent
            border.width: 1.5
            radius: 2
        }
    }

    Row {
        anchors.left: map.left
        anchors.right: map.right
        anchors.top: map.bottom
        anchors.topMargin: 6
        Text { text: "ズーム"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 12; width: parent.width / 2 }
        Text {
            width: parent.width / 2
            horizontalAlignment: Text.AlignRight
            text: Math.round(root.zoom * 100) + "%"
            color: Theme.text; font.family: Theme.mono; font.pixelSize: 12
        }
    }
}
