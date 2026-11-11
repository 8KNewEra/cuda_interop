import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 画像処理パネル（モックの右ドロワー）。
//   フィルタは配列 1 本で持つ。追加するときは filters に 1 行足し、
//   PlayerController に同名の bool プロパティを生やす。
//   並びは FBO_Rendering の適用順（Gaussian → Sobel → Averaging）と合わせること。
Rectangle {
    id: root
    signal closeRequested()

    readonly property var filters: [
        { key: "gaussian",  name: "ガウシアンフィルタ", kind: "GLSL · ぼかし" },
        { key: "sobel",     name: "ソーベルフィルタ",   kind: "GLSL · エッジ検出" },
        { key: "averaging", name: "平均化フィルタ",     kind: "GLSL · 平滑化" }
    ]
    readonly property int activeCount: (player.gaussian ? 1 : 0) + (player.sobel ? 1 : 0) + (player.averaging ? 1 : 0)
    property string expandedKey: ""

    // 有効なものの中での適用順（1 始まり）。無効なら 0。
    function orderOf(key) {
        var n = 0
        for (var i = 0; i < filters.length; i++) {
            if (player[filters[i].key]) n++
            if (filters[i].key === key) return player[key] ? n : 0
        }
        return 0
    }

    width: 300
    radius: Theme.radius
    color: Theme.panel
    border.color: Theme.borderSoft
    clip: true

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ヘッダー
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 14
            Layout.leftMargin: 16
            Layout.bottomMargin: 10
            ColumnLayout {
                spacing: 2
                Text { text: "画像処理"; color: Theme.text; font.family: Theme.font; font.pixelSize: 14; font.bold: true }
                Text {
                    text: root.activeCount + " / " + root.filters.length + " 有効 · 上から順に適用"
                    color: Theme.muted; font.family: Theme.font; font.pixelSize: 12
                }
            }
            Item { Layout.fillWidth: true }
            IconButton {
                implicitWidth: 36; implicitHeight: 36
                iconSize: 16
                paths: ["M6 6l12 12M18 6L6 18"]
                tip: "閉じる"
                onClicked: root.closeRequested()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            Text { text: "フィルタ（GLSL）"; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 11; font.letterSpacing: 0.6 }
            Item { Layout.fillWidth: true }
            AbstractButton {
                id: offBtn
                text: "すべて OFF"
                enabled: player.controlsEnabled && root.activeCount > 0
                implicitHeight: 28
                implicitWidth: offLbl.implicitWidth + 16
                hoverEnabled: true
                focusPolicy: Qt.NoFocus
                onClicked: player.allFiltersOff()
                background: Rectangle { radius: 6; color: offBtn.hovered && offBtn.enabled ? Theme.hover : "transparent" }
                contentItem: Text {
                    id: offLbl
                    text: offBtn.text
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    color: offBtn.enabled ? Theme.muted : Theme.faint
                    font.family: Theme.font; font.pixelSize: 12
                }
            }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: list.implicitHeight + 24
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            ColumnLayout {
                id: list
                x: 12; y: 8
                width: parent.width - 24
                spacing: 8

                Repeater {
                    model: root.filters
                    delegate: Rectangle {
                        id: card
                        required property var modelData
                        readonly property bool on: player[modelData.key]
                        readonly property bool expanded: root.expandedKey === modelData.key
                        readonly property int  order: root.orderOf(modelData.key)

                        Layout.fillWidth: true
                        implicitHeight: cardCol.implicitHeight
                        radius: 10
                        color: Theme.panel2
                        border.color: on ? Qt.rgba(79/255, 209/255, 197/255, 0.45) : Theme.border

                        ColumnLayout {
                            id: cardCol
                            width: parent.width
                            spacing: 0

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.margins: 8
                                Layout.leftMargin: 12
                                spacing: 10

                                Rectangle {
                                    width: 22; height: 22; radius: 6
                                    color: card.on ? Theme.accent : Theme.border
                                    Text {
                                        anchors.centerIn: parent
                                        text: card.order > 0 ? card.order : "–"
                                        color: card.on ? Theme.accentInk : Theme.muted2
                                        font.family: Theme.mono; font.pixelSize: 12
                                    }
                                }

                                // 名前（クリックでパラメータ欄を開閉）
                                MouseArea {
                                    Layout.fillWidth: true
                                    implicitHeight: 36
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: root.expandedKey = card.expanded ? "" : card.modelData.key
                                    RowLayout {
                                        anchors.fill: parent
                                        spacing: 6
                                        ColumnLayout {
                                            spacing: 0
                                            Layout.fillWidth: true
                                            Text { text: card.modelData.name; color: card.on ? Theme.text : "#AEB4BD"; font.family: Theme.font; font.pixelSize: 13; font.weight: Font.Medium }
                                            Text { text: card.modelData.kind; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 11 }
                                        }
                                        Icon {
                                            size: 14
                                            color: Theme.muted2
                                            paths: ["M9 6l6 6-6 6"]
                                            rotation: card.expanded ? 90 : 0
                                            Behavior on rotation { NumberAnimation { duration: 120 } }
                                        }
                                    }
                                }

                                ToggleSwitch {
                                    checked: card.on
                                    enabled: player.controlsEnabled
                                    onToggled: player[card.modelData.key] = checked
                                }
                            }

                            Rectangle {
                                visible: card.expanded
                                Layout.fillWidth: true
                                Layout.leftMargin: 12
                                Layout.rightMargin: 12
                                Layout.bottomMargin: 12
                                implicitHeight: paramText.implicitHeight + 20
                                radius: 8
                                color: "#0F1113"
                                border.color: Theme.border
                                Text {
                                    id: paramText
                                    anchors.fill: parent
                                    anchors.margins: 10
                                    wrapMode: Text.WordWrap
                                    text: "パラメータ領域。現在は調整項目なし（カーネルサイズや強度などを追加する場合はここに並べる）"
                                    color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12
                                    lineHeight: 1.3
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    implicitHeight: 40
                    radius: 10
                    color: "transparent"
                    border.color: Theme.border
                    Text {
                        anchors.centerIn: parent
                        text: "今後追加する処理（CUDA / GLSL）はここに並ぶ"
                        color: "#6F7782"; font.family: Theme.font; font.pixelSize: 12
                    }
                }
            }
        }
    }
}
