import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 係数の直接編集（ウィンドウ中央に出すポップアップ）
//   セルに数値を入れて Enter / Tab / フォーカスアウトで確定 → すぐ映像に反映
//   Tab / Shift+Tab・↑↓ でセル移動。Esc で閉じる（入力中のセルは取り消し）
Popup {
    id: root
    property var kernel: null
    property string title: ""
    readonly property int n: kernel ? kernel.size : 0
    readonly property real cellW: 72
    readonly property real cellH: 34

    parent: Overlay.overlay
    anchors.centerIn: parent
    modal: true
    focus: true
    padding: 0
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    Overlay.modal: Rectangle { color: Qt.rgba(0, 0, 0, 0.5) }

    function fmt(v) {
        // 小数 6 桁で丸めて末尾の 0 を落とす（0.002300 → 0.0023）
        return String(parseFloat(Number(v).toFixed(6)))
    }
    function focusCell(i) {
        if (i < 0 || i >= cells.count) return
        const c = cells.itemAt(i)
        if (c) { c.input.forceActiveFocus(); c.input.selectAll() }
    }

    background: Rectangle {
        radius: 14
        color: Theme.bg
        border.color: Theme.border
    }

    contentItem: ColumnLayout {
        spacing: 0

        // ヘッダー
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 18
            Layout.bottomMargin: 6
            spacing: 10
            Text {
                text: root.title + " — 係数（" + root.n + "×" + root.n + "）"
                color: Theme.text; font.family: Theme.font; font.pixelSize: 16; font.bold: true
            }
            Item { Layout.fillWidth: true }
            IconButton {
                implicitWidth: 34; implicitHeight: 34
                iconSize: 16
                paths: ["M6 6l12 12M18 6L6 18"]
                tip: "閉じる（Esc）"
                onClicked: root.close()
            }
        }
        Text {
            Layout.leftMargin: 18
            Layout.rightMargin: 18
            text: "サイズまたは σ を変更すると係数が再計算され、編集内容は破棄されます。"
            color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12
        }

        // 係数のマス
        Grid {
            id: grid
            Layout.margins: 18
            Layout.alignment: Qt.AlignHCenter
            columns: root.n
            spacing: 4
            Repeater {
                id: cells
                // 値が変わってもセルを作り直さない（入力中のフォーカスを保つ）ため、モデルは個数だけ
                model: root.n * root.n
                delegate: Rectangle {
                    id: cell
                    required property int index
                    readonly property real modelData: root.kernel ? (root.kernel.coeffs[index] ?? 0) : 0
                    property alias input: field
                    readonly property bool center: index === Math.floor(root.n * root.n / 2)
                    width: root.cellW; height: root.cellH
                    radius: 6
                    color: field.activeFocus ? Theme.hover : Theme.panel2
                    border.color: field.activeFocus ? Theme.accent : (center ? Theme.dashed : Theme.border)
                    TextInput {
                        id: field
                        anchors.fill: parent
                        anchors.leftMargin: 6
                        anchors.rightMargin: 6
                        horizontalAlignment: TextInput.AlignHCenter
                        verticalAlignment: TextInput.AlignVCenter
                        text: root.fmt(cell.modelData)
                        validator: RegularExpressionValidator { regularExpression: /-?\d*\.?\d*(e-?\d+)?/ }
                        color: cell.modelData < 0 ? Theme.marker : Theme.text
                        selectionColor: Theme.accent
                        selectByMouse: true
                        clip: true
                        font.family: Theme.mono; font.pixelSize: 12
                        function commit() {
                            const v = parseFloat(text)
                            // 表示どおりなら書き戻さない（Tab で通過しただけで丸めた値に変えない）
                            if (!isNaN(v) && text !== root.fmt(cell.modelData)) root.kernel.setCoeff(cell.index, v)
                            text = Qt.binding(() => root.fmt(cell.modelData))
                        }
                        onEditingFinished: commit()
                        onActiveFocusChanged: if (activeFocus) selectAll()
                        Keys.onTabPressed:     { commit(); root.focusCell(cell.index + 1) }
                        Keys.onBacktabPressed: { commit(); root.focusCell(cell.index - 1) }
                        Keys.onUpPressed:      { commit(); root.focusCell(cell.index - root.n) }
                        Keys.onDownPressed:    { commit(); root.focusCell(cell.index + root.n) }
                    }
                }
            }
        }

        // フッター
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 60
            color: Theme.bar
            radius: 14
            // 上側の角は四角に（下だけ丸める）
            Rectangle { anchors.top: parent.top; width: parent.width; height: 14; color: Theme.bar }
            Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Theme.borderSoft }
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: 12
                Text {
                    text: "合計 " + (root.kernel ? root.kernel.sum.toFixed(5) : "-")
                    color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                }
                Item { Layout.fillWidth: true }
                Text { text: "正規化"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 12 }
                ToggleSwitch {
                    checked: root.kernel ? root.kernel.normalize : false
                    onToggled: root.kernel.setNormalize(checked)
                }
                Rectangle { width: 1; height: 20; color: Theme.border }
                AbstractButton {
                    id: resetBtn
                    text: "リセット"
                    enabled: root.kernel && root.kernel.custom
                    implicitHeight: 32
                    implicitWidth: 84
                    hoverEnabled: true
                    focusPolicy: Qt.NoFocus
                    opacity: enabled ? 1 : 0.4
                    onClicked: root.kernel.reset()
                    background: Rectangle { radius: 8; color: resetBtn.hovered && resetBtn.enabled ? Theme.hover : Theme.panel2; border.color: Theme.border }
                    contentItem: Text {
                        text: resetBtn.text
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        color: Theme.text2; font.family: Theme.font; font.pixelSize: 13
                    }
                }
            }
        }
    }
}
