import QtQuick
import QtQuick.Shapes

// 24x24 の SVG パスを線で描くアイコン
//   paths  : SVG の d 属性（最大 3 本）
//   filled : true なら塗り（再生/一時停止など）
Item {
    id: root
    property var    paths: []
    property color  color: "#C9CED6"
    property real   size: 18
    property real   strokeWidth: 2
    property bool   filled: false

    implicitWidth: size
    implicitHeight: size

    Shape {
        anchors.centerIn: parent
        width: 24; height: 24
        scale: root.size / 24
        preferredRendererType: Shape.CurveRenderer

        ShapePath {
            strokeColor: root.filled ? "transparent" : root.color
            fillColor:   root.filled ? root.color : "transparent"
            strokeWidth: root.strokeWidth
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin
            PathSvg { path: root.paths.length > 0 ? root.paths[0] : "" }
        }
        ShapePath {
            strokeColor: root.filled ? "transparent" : root.color
            fillColor:   root.filled ? root.color : "transparent"
            strokeWidth: root.strokeWidth
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin
            PathSvg { path: root.paths.length > 1 ? root.paths[1] : "" }
        }
        ShapePath {
            strokeColor: root.filled ? "transparent" : root.color
            fillColor:   root.filled ? root.color : "transparent"
            strokeWidth: root.strokeWidth
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin
            PathSvg { path: root.paths.length > 2 ? root.paths[2] : "" }
        }
    }
}
