import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// パネル用の「ラベル ＋ スライダー ＋ 数値欄」1 行（画像処理パネルの共通部品）
RowLayout {
    id: root
    property string label: ""
    property int from: 0
    property int to: 100
    property int value: 0
    property string unit: ""
    property int decimals: 0          // 小数表示（value は 10^decimals 倍の整数）
    property var defaultValue: undefined   // ダブルクリックでこの値に戻す（未指定なら何もしない）
    property int labelWidth: 64
    // 数値欄の幅。-1 なら桁数から自動。並べるときは同じ値を渡すとスライダーの長さがそろう
    property int fieldWidth: -1
    signal moved(int v)

    spacing: 6
    opacity: enabled ? 1.0 : 0.45

    Text { text: root.label; Layout.preferredWidth: root.labelWidth; color: Theme.muted; font.family: Theme.font; font.pixelSize: 12; elide: Text.ElideRight }
    Slider {
        id: s
        Layout.fillWidth: true
        from: root.from; to: root.to; stepSize: 1
        snapMode: Slider.SnapAlways
        value: root.value
        onMoved: root.moved(Math.round(value))
        background: Rectangle {
            x: s.leftPadding; y: s.topPadding + s.availableHeight / 2 - 2
            width: s.availableWidth; height: 4; radius: 2
            color: Theme.border
            Rectangle { width: s.visualPosition * parent.width; height: 4; radius: 2; color: Theme.accent }
        }
        handle: Rectangle {
            x: s.leftPadding + s.visualPosition * (s.availableWidth - width)
            y: s.topPadding + s.availableHeight / 2 - height / 2
            width: 14; height: 14; radius: 7
            color: "#FFFFFF"
        }
        // ダブルクリックで既定値に戻す
        TapHandler {
            onDoubleTapped: if (root.defaultValue !== undefined && root.defaultValue !== root.value) root.moved(root.defaultValue)
        }
        // ホイールで 1 ずつ
        WheelHandler {
            onWheel: (ev) => {
                const d = ev.angleDelta.y > 0 ? 1 : (ev.angleDelta.y < 0 ? -1 : 0)
                const v = Math.max(root.from, Math.min(root.to, root.value + d))
                if (v !== root.value) root.moved(v)
            }
        }
    }
    // 数値欄（直接入力できる小さなスピン）
    //   クリックして入力 → Enter / フォーカスアウトで確定、Esc で取り消し
    //   ↑↓・ホイール・右端の ▲▼ で 1 ずつ（Shift で 10。▲▼ は押しっぱなしで連続）
    Rectangle {
        id: box
        // 幅はいちばん長い値の桁数に合わせる（余白を作らずスライダーを広く）
        Layout.preferredWidth: root.fieldWidth > 0 ? root.fieldWidth : Math.ceil(longest.advanceWidth) + (root.unit.length > 0 ? unitMetrics.advanceWidth + 1 : 0) + 8 + 17
        TextMetrics {
            id: longest
            font.family: Theme.mono; font.pixelSize: 12
            // from / to のうち文字数の多い方（"-100" や "1.00" など）
            text: { const a = box.show(root.from), b = box.show(root.to); return a.length >= b.length ? a : b }
        }
        TextMetrics { id: unitMetrics; font.family: Theme.mono; font.pixelSize: 11; text: root.unit }
        implicitHeight: 24
        radius: 6
        color: field.activeFocus ? Theme.hover : Theme.panel2
        border.color: field.activeFocus ? Theme.accent : (boxHover.hovered ? Theme.muted2 : Theme.border)

        function show(v) { return root.decimals > 0 ? (v / Math.pow(10, root.decimals)).toFixed(root.decimals) : String(v) }
        function stepBy(n) {
            const v = Math.max(root.from, Math.min(root.to, root.value + n))
            if (v !== root.value) root.moved(v)
            field.text = Qt.binding(() => box.show(root.value))
        }

        HoverHandler { id: boxHover; cursorShape: Qt.IBeamCursor }
        TapHandler { onTapped: field.forceActiveFocus() }   // 枠のどこを押しても入力できる
        WheelHandler {
            onWheel: (ev) => {
                const n = (ev.modifiers & Qt.ShiftModifier) ? 10 : 1
                const d = ev.angleDelta.y !== 0 ? ev.angleDelta.y : ev.angleDelta.x
                if (d > 0) box.stepBy(n); else if (d < 0) box.stepBy(-n)
            }
        }

        // ▲▼（縦に 2 つ。幅 16px）
        Column {
            id: arrows
            anchors.right: parent.right
            anchors.rightMargin: 1
            anchors.verticalCenter: parent.verticalCenter
            // inline component ではなく Repeater（外側の id を確実に参照するため）
            Repeater {
                model: [true, false]          // ▲, ▼
                delegate: AbstractButton {
                    id: ab
                    required property bool modelData
                    readonly property bool up: modelData
                    width: 16
                    height: (box.height - 2) / 2
                    autoRepeat: true
                    autoRepeatDelay: 350
                    autoRepeatInterval: 50
                    focusPolicy: Qt.NoFocus
                    hoverEnabled: true
                    enabled: up ? root.value < root.to : root.value > root.from
                    onClicked: box.stepBy(up ? 1 : -1)
                    background: Rectangle { color: ab.down ? Theme.selected : (ab.hovered && ab.enabled ? Theme.hover : "transparent") }
                    contentItem: Item {
                        Icon {
                            anchors.centerIn: parent
                            size: 10
                            color: ab.enabled ? Theme.text2 : Theme.faint
                            paths: [ab.up ? "M6 15l6-6 6 6" : "M6 9l6 6 6-6"]
                        }
                    }
                }
            }
        }
        Rectangle { anchors.right: arrows.left; width: 1; height: parent.height - 2; anchors.verticalCenter: parent.verticalCenter; color: Theme.border }

        Row {
            anchors.right: arrows.left
            anchors.rightMargin: 4
            anchors.verticalCenter: parent.verticalCenter
            spacing: 1
            TextInput {
                id: field
                width: box.width - 17 - 8 - (root.unit.length > 0 ? unitText.implicitWidth + 1 : 0)
                horizontalAlignment: TextInput.AlignRight
                text: box.show(root.value)
                validator: RegularExpressionValidator { regularExpression: root.decimals > 0 ? /-?\d*\.?\d*/ : /-?\d*/ }
                color: Theme.text2
                selectionColor: Theme.accent
                selectByMouse: true
                clip: true
                font.family: Theme.mono; font.pixelSize: 12
                function commit() {
                    const v = Math.round(parseFloat(text) * Math.pow(10, root.decimals))
                    if (!isNaN(v)) {
                        const c = Math.max(root.from, Math.min(root.to, v))
                        if (c !== root.value) root.moved(c)
                    }
                    text = Qt.binding(() => box.show(root.value))
                }
                onEditingFinished: commit()
                onActiveFocusChanged: if (activeFocus) selectAll()
                Keys.onUpPressed:   (e) => box.stepBy((e.modifiers & Qt.ShiftModifier) ? 10 : 1)
                Keys.onDownPressed: (e) => box.stepBy((e.modifiers & Qt.ShiftModifier) ? -10 : -1)
                Keys.onEscapePressed: { text = Qt.binding(() => box.show(root.value)); focus = false }
            }
            Text {
                id: unitText
                visible: root.unit.length > 0
                text: root.unit
                color: Theme.muted; font.family: Theme.mono; font.pixelSize: 11
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }
}
