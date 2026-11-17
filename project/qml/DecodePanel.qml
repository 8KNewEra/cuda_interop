import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// デコード設定（マルチ GPU デコード）。画面全体に重ねるダイアログ（画像保存設定と同じ見た目）
//   値は DecodeSettings（context property "decodeSettings"）が持つ
//   GPU ごとの「デコードの重み」でタイルを按分する（decode_gpu_map.h）。0 の GPU はデコードに使わない
//   デコーダはファイルを開くときに作るので、反映は次に開いたとき。「OK して開き直す」なら今の位置から開き直す
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
                    text: "タイル分割された動画のタイルを、重みの比で複数の GPU に振り分けてデコードします（マルチ GPU デコード）。"
                          + "重み 0 の GPU は使いません。表示に使っている GPU 以外でデコードしたタイルは、表示用の GPU へ転送します。"
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

                    // 行（配列は変更のたびに作り直されるので、モデルは本数にして中身は index で読む）
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
                                          + (cell.g.canDecode === false ? " · このコーデックは非対応" : "")
                                    color: cell.g.canDecode === false ? Theme.marker : Theme.faint
                                    font.family: Theme.font; font.pixelSize: 11
                                }
                            }
                            // NVDEC 使用率（開いた時点の値）
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
                            // このファイルで受け持つタイル数の見込み
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
                          ? "今のファイルは " + decodeSettings.fileTiles + " タイルです。設定はファイルを開き直すと反映されます。"
                          : "タイル分割の動画を開いているときは、ここにタイルの割り振りの見込みが出ます。設定は次にファイルを開いたときから使われます。"
                    color: Theme.faint; font.family: Theme.font; font.pixelSize: 12
                }

                Rectangle { Layout.fillWidth: true; Layout.topMargin: 4; height: 1; color: Theme.borderSoft }

                // 転送の圧縮（試験的。nvCOMP）。別 GPU でデコードしたタイルを圧縮してから表示用 GPU へ送る
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Text { text: "転送の圧縮（試験的）"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 13; Layout.preferredWidth: 160 }
                    Select {
                        Layout.fillWidth: true
                        tip: "別の GPU でデコードしたタイルを、圧縮してから表示用の GPU へ送ります。GPU 間の経路が細い構成で効果があります。性能 HUD の「圧縮率」「転送」「展開」で効果を確かめ、効果がなければ「なし」に戻してください。"
                        options: decodeSettings.xferCodecOptions
                        value: decodeSettings.xferCodec
                        onPicked: (v) => decodeSettings.setXferCodec(v)
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    visible: decodeSettings.xferCodec !== "none"
                    Text { text: "差分（前処理）"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 13; Layout.preferredWidth: 160 }
                    Select {
                        Layout.fillWidth: true
                        tip: "圧縮の前に、行の中で隣の画素との差に置き換えます。なめらかな映像ほど縮みやすくなります。"
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
