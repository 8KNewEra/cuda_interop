import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 色補正の設定（彩度・色温度・ティント ＋ R / G / B ごとのゲイン・オフセット・ガンマ）
//   チャンネルはタブで切り替え、選んだチャンネルの 3 本だけを表示する。
//   スライダーはダブルクリックで既定値、数値欄は直接入力・▲▼・ホイール（共通の MiniSlider）
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    property int ch: 0                                   // 0 R / 1 G / 2 B
    readonly property var chKey: ["R", "G", "B"]
    readonly property var chColor: ["#FF5A52", "#46D369", "#4A9DFF"]

    function changedCh(i) {
        const k = chKey[i]
        return p["gain" + k] !== 100 || p["offset" + k] !== 0 || p["gamma" + k] !== 100
    }
    readonly property bool changed: p.saturation !== 100 || p.temperature !== 0 || p.tint !== 0 || changedCh(0) || changedCh(1) || changedCh(2)

    spacing: 8

    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "彩度"
        from: 0; to: 200; unit: "%"; defaultValue: 100
        value: root.p.saturation
        onMoved: (v) => player.setFilterParam("saturation", v)
    }

    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "色温度"
        from: -100; to: 100; defaultValue: 0
        value: root.p.temperature
        onMoved: (v) => player.setFilterParam("temperature", v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "ティント"
        from: -100; to: 100; defaultValue: 0
        value: root.p.tint
        onMoved: (v) => player.setFilterParam("tint", v)
    }

    Rectangle { Layout.fillWidth: true; Layout.topMargin: 2; Layout.bottomMargin: 2; height: 1; color: Theme.borderSoft }

    // ---- チャンネルのタブ（変更があれば右上に点）----
    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 30
        radius: 7
        color: Theme.panel2
        border.color: Theme.border
        opacity: enabled ? 1.0 : 0.45
        Row {
            anchors.fill: parent
            anchors.margins: 2
            Repeater {
                model: 3
                delegate: AbstractButton {
                    id: tab
                    required property int index
                    readonly property bool on: root.ch === index
                    width: parent.width / 3
                    height: parent.height
                    hoverEnabled: true
                    focusPolicy: Qt.NoFocus
                    onClicked: root.ch = index
                    background: Rectangle { radius: 5; color: tab.on ? Theme.selected : (tab.hovered ? Theme.hover : "transparent") }
                    contentItem: Item {
                        Row {
                            anchors.centerIn: parent
                            spacing: 6
                            Rectangle { width: 8; height: 8; radius: 4; color: root.chColor[tab.index]; anchors.verticalCenter: parent.verticalCenter }
                            Text {
                                text: root.chKey[tab.index]
                                color: tab.on ? "#FFFFFF" : Theme.muted2
                                font.family: Theme.mono; font.pixelSize: 12; font.bold: tab.on
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        // 既定値から変えたチャンネルの印
                        Rectangle {
                            visible: root.changedCh(tab.index)
                            anchors.right: parent.right; anchors.top: parent.top
                            anchors.rightMargin: 6; anchors.topMargin: 5
                            width: 5; height: 5; radius: 2.5
                            color: Theme.marker
                        }
                    }
                }
            }
        }
    }

    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "ゲイン"
        from: 0; to: 200; unit: "%"; defaultValue: 100
        value: root.p["gain" + root.chKey[root.ch]]
        onMoved: (v) => player.setFilterParam("gain" + root.chKey[root.ch], v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "オフセット"
        from: -100; to: 100; defaultValue: 0
        value: root.p["offset" + root.chKey[root.ch]]
        onMoved: (v) => player.setFilterParam("offset" + root.chKey[root.ch], v)
    }
    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 72
        fieldWidth: 58
        label: "ガンマ"
        from: 10; to: 300; decimals: 2; defaultValue: 100
        value: root.p["gamma" + root.chKey[root.ch]]
        onMoved: (v) => player.setFilterParam("gamma" + root.chKey[root.ch], v)
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.topMargin: 2
        spacing: 8
        Text {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: "ゲインは明部、オフセットは暗部の色かぶり補正に使用します。"
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
            onClicked: player.resetColorCorrect()
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
