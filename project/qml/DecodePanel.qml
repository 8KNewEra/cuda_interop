import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// デコード設定（マルチ GPU デコード）。値は DecodeSettings（"decodeSettings"）。docs/MULTI_GPU_DECODE.md
Item {
    id: root

    component PlainButton: AbstractButton {
        id: pb
        property bool primary: false
        implicitHeight: 40
        implicitWidth: Math.max(112, pbText.implicitWidth + 40)
        hoverEnabled: true
        focusPolicy: Qt.NoFocus
        opacity: enabled ? 1.0 : 0.4
        background: Rectangle {
            radius: 10
            color: pb.primary ? (pb.hovered ? Qt.lighter(Theme.accent, 1.08) : Theme.accent)
                              : (pb.hovered ? Theme.hover : Theme.panel2)
            border.width: pb.primary ? 0 : 1
            border.color: Theme.border
        }
        contentItem: Text {
            id: pbText
            text: pb.text
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            color: pb.primary ? Theme.accentInk : Theme.text2
            font.family: Theme.font
            font.pixelSize: 14
            font.weight: pb.primary ? Font.Bold : Font.Medium
        }
    }

    component HeadText: Text { color: Theme.faint; font.family: Theme.font; font.pixelSize: 12 }

    readonly property bool hasFile: decodeSettings.fileTiles > 0

    // ---- 暗幕（後ろの再生画面は触れない）----
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.6)
        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.AllButtons
            onWheel: (w) => w.accepted = true
        }
    }

    Keys.onEscapePressed: decodeSettings.cancel()

    // ---- 本体 ----
    Rectangle {
        id: sheet
        anchors.centerIn: parent
        width: Math.min(parent.width - 48, 640)
        height: Math.min(parent.height - 48, col.implicitHeight)
        radius: 14
        color: Theme.bg
        border.color: Theme.border
        clip: true

        ColumnLayout {
            id: col
            anchors.left: parent.left
            anchors.right: parent.right
            spacing: 0

            // ヘッダー
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 60
                color: Theme.bar
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.borderSoft }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 12
                    spacing: 12
                    Icon {
                        size: 22; color: Theme.accent
                        paths: ["M4 6h7v5H4zM13 6h7v5h-7zM4 13h7v5H4zM13 13h7v5h-7z"]
                    }
                    Text { text: "デコード設定"; color: Theme.text; font.family: Theme.font; font.pixelSize: 18; font.bold: true }
                    Item { Layout.fillWidth: true }
                    IconButton {
                        implicitWidth: 40; implicitHeight: 40
                        iconSize: 18
                        paths: ["M6 6l12 12M18 6L6 18"]
                        tip: "閉じる（Esc）"
                        onClicked: decodeSettings.cancel()
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.margins: 24
                Layout.topMargin: 18
                Layout.bottomMargin: 20
                spacing: 12

                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    text: "タイルを重みの比で GPU に割り振ってデコードします。重み 0 の GPU は使いません。"
                    color: Theme.muted; font.family: Theme.font; font.pixelSize: 13
                    lineHeight: 1.25
                }

                // 表
                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 16
                    rowSpacing: 8

                    HeadText { text: "GPU"; Layout.fillWidth: true }
                    HeadText { text: "NVDEC"; Layout.preferredWidth: 56; horizontalAlignment: Text.AlignRight }
                    HeadText { text: "重み"; Layout.preferredWidth: 150 }
                    HeadText { text: root.hasFile ? "このファイル" : ""; Layout.preferredWidth: 84; horizontalAlignment: Text.AlignRight }

                    // 配列は変更のたびに作り直されるので、モデルは本数にして中身は index で読む
                    Repeater {
                        model: decodeSettings.gpus.length * 4
                        delegate: Item {
                            id: cell
                            required property int index
                            readonly property var g: decodeSettings.gpus[Math.floor(index / 4)] || ({})
                            readonly property int col: index % 4
                            Layout.fillWidth: col === 0
                            Layout.preferredWidth: [0, 56, 150, 84][col]
                            implicitHeight: 34

                            // GPU 名
                            Column {
                                visible: cell.col === 0
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width
                                Text {
                                    width: parent.width
                                    text: (cell.g.name || "") + (cell.g.primary ? "（表示用）" : "")
                                    elide: Text.ElideRight
                                    color: Theme.text2; font.family: Theme.font; font.pixelSize: 13
                                }
                                Text {
                                    text: "GPU " + cell.g.id + " · " + (cell.g.sm || "")
                                          + (cell.g.canDecode === false ? " · 非対応のコーデック" : "")
                                    color: cell.g.canDecode === false ? Theme.marker : Theme.faint
                                    font.family: Theme.font; font.pixelSize: 11
                                }
                            }
                            Text {
                                visible: cell.col === 1
                                anchors.fill: parent
                                horizontalAlignment: Text.AlignRight
                                verticalAlignment: Text.AlignVCenter
                                text: (cell.g.dec !== undefined ? cell.g.dec : "—") + " %"
                                color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                            }
                            // 重み
                            NumberSpin {
                                visible: cell.col === 2
                                anchors.verticalCenter: parent.verticalCenter
                                width: 150
                                from: 0; to: 10
                                value: cell.g.weight || 0
                                enabled: cell.g.canDecode !== false
                                onMoved: (v) => decodeSettings.setWeight(cell.g.id, v)
                            }
                            Text {
                                visible: cell.col === 3
                                anchors.fill: parent
                                horizontalAlignment: Text.AlignRight
                                verticalAlignment: Text.AlignVCenter
                                text: cell.g.tiles >= 0 ? cell.g.tiles + " タイル" : ""
                                color: cell.g.tiles > 0 ? Theme.text : Theme.faint
                                font.family: Theme.font; font.pixelSize: 13
                                font.weight: cell.g.tiles > 0 ? Font.Medium : Font.Normal
                            }
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    text: root.hasFile
                          ? "このファイル: " + decodeSettings.fileTiles + " タイル。開き直すと反映されます。"
                          : "次に開いたファイルから反映されます。"
                    color: Theme.faint; font.family: Theme.font; font.pixelSize: 12
                }

                Rectangle { Layout.fillWidth: true; Layout.topMargin: 4; height: 1; color: Theme.borderSoft }

                // 転送の圧縮（GPU が 1 枚なら転送が無いので「なし」表示で触れない。保存値はそのまま）
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Text { text: "転送の圧縮"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 13; Layout.preferredWidth: 120 }
                    Select {
                        Layout.fillWidth: true
                        enabled: decodeSettings.multiGpu
                        tip: "別の GPU から表示用 GPU へ送るタイルを圧縮します（可逆）。"
                        options: decodeSettings.xferCodecOptions
                        value: decodeSettings.multiGpu ? decodeSettings.xferCodec : "none"
                        onPicked: (v) => decodeSettings.setXferCodec(v)
                    }
                }
                Text {
                    visible: !decodeSettings.multiGpu
                    text: "GPU が 1 枚のときは転送しないので使いません。"
                    color: Theme.faint; font.family: Theme.font; font.pixelSize: 12
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    visible: decodeSettings.multiGpu && decodeSettings.xferCodec !== "none"
                    Text { text: "差分"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 13; Layout.preferredWidth: 120 }
                    Select {
                        Layout.fillWidth: true
                        tip: "左隣との差を取ってから圧縮します。圧縮率が上がります。"
                        options: [{ text: "あり", value: true }, { text: "なし", value: false }]
                        value: decodeSettings.xferDelta
                        onPicked: (v) => decodeSettings.setXferDelta(v)
                    }
                }
            }

            // フッター
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 72
                color: Theme.bar
                Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Theme.borderSoft }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 12
                    Item { Layout.fillWidth: true }
                    PlainButton { text: "キャンセル"; onClicked: decodeSettings.cancel() }
                    PlainButton {
                        visible: root.hasFile
                        enabled: decodeSettings.dirty
                        text: "OK して開き直す"
                        onClicked: decodeSettings.apply(true)
                    }
                    PlainButton { primary: true; text: "OK"; onClicked: decodeSettings.apply(false) }
                }
            }
        }
    }
}
