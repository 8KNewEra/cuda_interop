import QtQuick

// 再生位置＋再生範囲のスライダー
//   ・トラックをクリック／ドラッグ → scrubbed（操作中）→ released（離したとき）
//   ・橙のつまみ → 開始 / 終了を動かす
//   つまみは開始＝左、終了＝右に張り出したタブ。3 点が重なっても
//   左半分を掴めば開始、右半分なら終了、トラック上なら再生位置になる
Item {
    id: root
    property int  maximum: 1000
    property int  value: 0
    property int  rangeStart: 0
    property int  rangeEnd: 1000
    property bool interactive: true

    signal scrubbed(int v)
    signal released(int v)
    signal startMoved(int v)
    signal endMoved(int v)
    signal rangeReleased()

    // ドラッグ中はつまみ位置をここで持つ（デコード結果を待たずに追従）
    property bool dragging: playArea.pressed
    property int  dragValue: 0
    // 再生位置は常に範囲内に描く（範囲つまみで押し出された直後も追従）
    readonly property int shownValue: Math.max(rangeStart, Math.min(rangeEnd, dragging ? dragValue : value))

    // レイアウト（縦）
    readonly property real trackY: 14          // トラック中心
    readonly property real tabTop: 22          // つまみのタブ上端
    readonly property real tabH: 18
    readonly property real tabW: 18

    implicitHeight: tabTop + tabH + 2
    opacity: interactive ? 1.0 : 0.45

    function xFor(v) { return maximum > 0 ? width * Math.max(0, Math.min(1, v / maximum)) : 0 }
    function vFor(x) { return maximum > 0 ? Math.round(Math.max(0, Math.min(1, x / width)) * maximum) : 0 }
    function vInRange(x) { return Math.max(rangeStart, Math.min(rangeEnd, vFor(x))) }

    // トラック
    Rectangle { x: 0; width: parent.width; y: root.trackY - 2; height: 4; radius: 2; color: Theme.border }
    // 再生範囲
    Rectangle {
        x: root.xFor(root.rangeStart); width: Math.max(2, root.xFor(root.rangeEnd) - x)
        y: root.trackY - 3; height: 6; radius: 3
        color: Qt.rgba(242/255, 165/255, 65/255, 0.25)
    }
    // 再生済み（範囲の始端から現在位置まで）
    Rectangle {
        x: root.xFor(root.rangeStart)
        width: Math.max(0, root.xFor(root.shownValue) - x)
        y: root.trackY - 3; height: 6; radius: 3
        color: Theme.accent
    }

    // 再生位置のドラッグ（トラック全体）
    MouseArea {
        id: playArea
        x: 0; y: 0
        width: parent.width
        height: root.tabTop
        enabled: root.interactive
        cursorShape: Qt.PointingHandCursor
        onPressed: (mouse) => { root.dragValue = root.vInRange(mouse.x); root.scrubbed(root.dragValue) }
        onPositionChanged: (mouse) => {
            if (!pressed) return
            const v = root.vInRange(mouse.x)
            if (v !== root.dragValue) { root.dragValue = v; root.scrubbed(v) }
        }
        onReleased: (mouse) => root.released(root.vInRange(mouse.x))
    }

    // 開始 / 終了のつまみ
    //   side = -1: 開始（タブは左へ）、+1: 終了（タブは右へ）
    component Marker: Item {
        id: m
        property int  frame
        property int  side: -1
        signal moved(int v)
        width: root.width
        height: root.height

        readonly property real px: root.xFor(frame)
        readonly property bool hot: grip.containsMouse || grip.pressed

        // 位置を示す縦線（トラックを貫く）
        Rectangle {
            x: m.px - 1
            y: root.trackY - 9
            width: 2
            height: root.tabTop - (root.trackY - 9) + 2
            color: Theme.marker
        }
        // 掴むためのタブ（外側へ張り出す）
        Rectangle {
            id: tab
            x: m.side < 0 ? m.px - root.tabW : m.px - 1
            y: root.tabTop
            width: root.tabW + 1
            height: root.tabH
            radius: 4
            color: m.hot ? Qt.lighter(Theme.marker, 1.15) : Theme.marker
            // 角を 1 つだけ立てて、どちら向きのつまみか分かるようにする
            Rectangle {
                width: 6; height: 6
                color: parent.color
                x: m.side < 0 ? parent.width - width : 0
                y: 0
            }
            // グリップの溝
            Row {
                anchors.centerIn: parent
                spacing: 2
                Repeater {
                    model: 2
                    Rectangle { width: 1.5; height: 7; radius: 0.75; color: Qt.rgba(0, 0, 0, 0.45) }
                }
            }
        }

        // 当たり判定はタブより広く、縦線の外側半分まで含める
        MouseArea {
            id: grip
            x: m.side < 0 ? m.px - root.tabW - 6 : m.px
            y: root.trackY - 10
            width: root.tabW + 6
            height: root.tabTop + root.tabH + 4 - y
            enabled: root.interactive
            hoverEnabled: true
            cursorShape: Qt.SizeHorCursor
            preventStealing: true
            property real grabOffset: 0
            onPressed: (mouse) => {
                // 掴んだ点と線のずれを覚えておき、つまみが指から逃げないようにする
                const p = mapToItem(root, mouse.x, mouse.y)
                grabOffset = p.x - m.px
            }
            onPositionChanged: (mouse) => {
                if (!pressed) return
                const p = mapToItem(root, mouse.x, mouse.y)
                m.moved(root.vFor(p.x - grabOffset))
            }
            onReleased: root.rangeReleased()
        }
    }

    // 再生ヘッド（つまみの線より手前に描く）
    Rectangle {
        x: root.xFor(root.shownValue) - 8
        y: root.trackY - 8; width: 16; height: 16; radius: 8
        color: "#FFFFFF"
        border.width: 4
        border.color: Qt.rgba(79/255, 209/255, 197/255, 0.30)
        z: 1
    }

    Marker { side: -1; frame: root.rangeStart; onMoved: (v) => root.startMoved(Math.min(v, root.rangeEnd)) }
    Marker { side:  1; frame: root.rangeEnd;   onMoved: (v) => root.endMoved(Math.max(v, root.rangeStart)) }
}
