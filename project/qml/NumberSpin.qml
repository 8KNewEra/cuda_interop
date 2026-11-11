import QtQuick
import QtQuick.Controls

// スピンボックス [ − | 値 単位 | ＋ ]
//   ・− / ＋ は押しっぱなしで連続
//   ・数値の上でホイール、↑↓ キー（Shift で 10 倍）
//   ・直接入力も可（Enter / フォーカスアウトで確定、Esc で取り消し）
//   値の本流は呼び出し側。moved(v) を出し、範囲を通った結果が value に戻る。
Rectangle {
    id: root
    property int    value: 0
    property int    from: 0
    property int    to: 100
    property int    stepSize: 1
    property string unit: ""
    signal moved(int v)

    function stepBy(n) {
        const v = Math.max(from, Math.min(to, value + n * stepSize))
        if (v !== value) moved(v)
        field.text = Qt.binding(() => root.value)
    }

    implicitWidth: 168
    implicitHeight: 34
    radius: 8
    color: Theme.panel2
    border.color: field.activeFocus ? Theme.accent : Theme.border
    opacity: enabled ? 1.0 : 0.45
    clip: true

    component SpinButton: AbstractButton {
        id: sb
        property var paths: []
        width: 30
        height: parent.height
        autoRepeat: true
        autoRepeatDelay: 350
        autoRepeatInterval: 50
        focusPolicy: Qt.NoFocus
        hoverEnabled: true
        background: Rectangle { color: sb.down ? Theme.selected : (sb.hovered && sb.enabled ? Theme.hover : "transparent") }
        contentItem: Item {
            Icon {
                anchors.centerIn: parent
                size: 13
                paths: sb.paths
                color: sb.enabled ? Theme.text2 : Theme.faint
            }
        }
    }

    Row {
        anchors.fill: parent
        anchors.margins: 1

        SpinButton {
            paths: ["M6 12h12"]
            enabled: root.value > root.from
            onClicked: root.stepBy(-1)
        }
        Rectangle { width: 1; height: parent.height; color: Theme.border }

        Item {
            width: parent.width - 2 * 30 - 2
            height: parent.height
            Row {
                anchors.centerIn: parent
                spacing: 4
                TextInput {
                    id: field
                    anchors.verticalCenter: parent.verticalCenter
                    width: Math.max(20, contentWidth)
                    horizontalAlignment: TextInput.AlignRight
                    text: root.value
                    validator: IntValidator { bottom: Math.min(0, root.from); top: Math.max(0, root.to) }
                    color: Theme.text
                    selectionColor: Theme.accent
                    selectByMouse: true
                    font.family: Theme.mono
                    font.pixelSize: 13
                    function commit() {
                        const v = parseInt(text)
                        if (!isNaN(v)) {
                            const c = Math.max(root.from, Math.min(root.to, v))
                            if (c !== root.value) root.moved(c)
                        }
                        text = Qt.binding(() => root.value)
                    }
                    onEditingFinished: commit()
                    Keys.onUpPressed:   (e) => root.stepBy((e.modifiers & Qt.ShiftModifier) ? 10 : 1)
                    Keys.onDownPressed: (e) => root.stepBy((e.modifiers & Qt.ShiftModifier) ? -10 : -1)
                    Keys.onEscapePressed: { text = Qt.binding(() => root.value); focus = false }
                }
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    visible: root.unit.length > 0
                    text: root.unit
                    color: Theme.muted
                    font.family: Theme.font
                    font.pixelSize: 12
                }
            }
            WheelHandler {
                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                onWheel: (ev) => {
                    const n = (ev.modifiers & Qt.ShiftModifier) ? 10 : 1
                    const d = ev.angleDelta.y !== 0 ? ev.angleDelta.y : ev.angleDelta.x
                    if (d > 0) root.stepBy(n)
                    else if (d < 0) root.stepBy(-n)
                }
            }
        }

        Rectangle { width: 1; height: parent.height; color: Theme.border }
        SpinButton {
            paths: ["M6 12h12", "M12 6v12"]
            enabled: root.value < root.to
            onClicked: root.stepBy(1)
        }
    }
}
