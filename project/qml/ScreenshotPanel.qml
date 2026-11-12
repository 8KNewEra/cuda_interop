import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 画像保存設定。画面全体に重ねるダイアログ（エンコード設定と同じ見た目）
//   値は ScreenshotSettings（context property "shotSettings"）が持つ
//   OK で反映、キャンセル / Esc で破棄、Enter は OK
Item {
    id: root

    // ---- 部品 ----
    component KeyLabel: Text {
        Layout.preferredWidth: 132
        color: Theme.muted
        font.family: Theme.font
        font.pixelSize: 13
        elide: Text.ElideRight
    }

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

    Keys.onEscapePressed: shotSettings.cancel()
    Keys.onReturnPressed: shotSettings.apply()
    Keys.onEnterPressed:  shotSettings.apply()

    // ---- 本体 ----
    Rectangle {
        id: sheet
        anchors.centerIn: parent
        width: Math.min(parent.width - 48, 600)
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
                        paths: ["M4 8a2 2 0 0 1 2-2h2l1.5-2h5L16 6h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z",
                                "M15.5 12.5a3.5 3.5 0 1 1-7 0a3.5 3.5 0 1 1 7 0z"]
                    }
                    Text { text: "画像保存設定"; color: Theme.text; font.family: Theme.font; font.pixelSize: 18; font.bold: true }
                    Item { Layout.fillWidth: true }
                    IconButton {
                        implicitWidth: 40; implicitHeight: 40
                        iconSize: 18
                        paths: ["M6 6l12 12M18 6L6 18"]
                        tip: "閉じる（Esc）"
                        onClicked: shotSettings.cancel()
                    }
                }
            }

            // 設定
            ColumnLayout {
                Layout.fillWidth: true
                Layout.margins: 24
                Layout.topMargin: 20
                Layout.bottomMargin: 20
                spacing: 10

                // 保存フォルダー
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    KeyLabel { text: "保存フォルダー" }
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 36
                        radius: 8
                        color: "#0F1113"
                        border.color: Theme.border
                        Text {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            verticalAlignment: Text.AlignVCenter
                            text: shotSettings.folder
                            elide: Text.ElideMiddle
                            color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                            ToolTip.visible: folderHover.containsMouse && truncated
                            ToolTip.delay: 400
                            ToolTip.text: shotSettings.folder
                            MouseArea { id: folderHover; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
                        }
                    }
                    PlainButton { text: "参照…"; implicitHeight: 36; onClicked: shotSettings.pickFolder() }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    KeyLabel { text: "保存フォーマット" }
                    Select {
                        Layout.fillWidth: true
                        options: shotSettings.formatOptions
                        value: shotSettings.format
                        onPicked: (v) => shotSettings.setValue("format", v)
                    }
                }

                RowLayout {
                    visible: shotSettings.showTiffCompression
                    Layout.fillWidth: true
                    spacing: 12
                    KeyLabel { text: "TIFF 圧縮" }
                    Select {
                        Layout.fillWidth: true
                        options: shotSettings.tiffCompressionOptions
                        value: shotSettings.tiffCompression
                        onPicked: (v) => shotSettings.setValue("tiffCompression", v)
                    }
                }

                ValueRow {
                    visible: shotSettings.showJpeg
                    Layout.fillWidth: true
                    label: "JPEG 品質"
                    from: 0; to: 100; stepSize: 1
                    value: shotSettings.jpegQuality
                    onMoved: (v) => shotSettings.setJpegQuality(v)
                }

                RowLayout {
                    visible: shotSettings.showJpeg
                    Layout.fillWidth: true
                    spacing: 12
                    KeyLabel { text: "色差サンプリング" }
                    Select {
                        Layout.fillWidth: true
                        tip: "4:4:4 は細部の色再現に優れますが、ファイルサイズが大きくなります。"
                        options: shotSettings.jpegSubsamplingOptions
                        value: shotSettings.jpegSubsampling
                        onPicked: (v) => shotSettings.setValue("jpegSubsampling", v)
                    }
                }

                RowLayout {
                    visible: shotSettings.showMode
                    Layout.fillWidth: true
                    spacing: 12
                    KeyLabel { text: "圧縮レベル" }
                    Select {
                        Layout.fillWidth: true
                        tip: "可逆圧縮のため画質は変わりません。high ほどファイルは小さくなり、保存に時間がかかります。"
                        options: shotSettings.modeOptions
                        value: shotSettings.mode
                        onPicked: (v) => shotSettings.setValue("mode", v)
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
                    PlainButton { text: "キャンセル"; onClicked: shotSettings.cancel() }
                    PlainButton { primary: true; text: "OK"; onClicked: shotSettings.apply() }
                }
            }
        }
    }
}
