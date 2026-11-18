import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 画像補正の設定（明るさ・コントラスト・シャドウ・ハイライト・ガンマ）
//   スライダーはダブルクリックでその項目だけ既定値に戻る
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    readonly property bool changed: p.brightness !== 0 || p.contrast !== 0 || p.shadows !== 0 || p.highlights !== 0 || p.gamma100 !== 100
    spacing: 8

    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "明るさ"
        from: -100; to: 100; defaultValue: 0
        value: root.p.brightness
        onMoved: (v) => player.setFilterParam("brightness", v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "コントラスト"
        from: -100; to: 100; defaultValue: 0
        value: root.p.contrast
        onMoved: (v) => player.setFilterParam("contrast", v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "シャドウ"
        from: -100; to: 100; defaultValue: 0
        value: root.p.shadows
        onMoved: (v) => player.setFilterParam("shadows", v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "ハイライト"
        from: -100; to: 100; defaultValue: 0
        value: root.p.highlights
        onMoved: (v) => player.setFilterParam("highlights", v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "ガンマ"
        from: 10; to: 300; decimals: 2; defaultValue: 100
        value: root.p.gamma100
        onMoved: (v) => player.setFilterParam("gamma100", v)
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.topMargin: 2
        spacing: 8
        Text {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: "スライダーをダブルクリックすると初期値に戻ります。"
            color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        }
        AbstractButton {
            id: resetBtn
            text: "初期値に戻す"
            enabled: root.changed
            implicitHeight: 28
            implicitWidth: resetLbl.implicitWidth + 20
            hoverEnabled: true
            focusPolicy: Qt.NoFocus
            opacity: enabled ? 1.0 : 0.4
            onClicked: player.resetImageAdjust()
            background: Rectangle { radius: 7; color: resetBtn.hovered && resetBtn.enabled ? Theme.hover : Theme.panel2; border.color: Theme.border }
            contentItem: Text {
                id: resetLbl
                text: resetBtn.text
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                color: Theme.text2; font.family: Theme.font; font.pixelSize: 12
            }
        }
    }
}
