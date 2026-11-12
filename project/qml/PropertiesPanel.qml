import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// プロパティ（動画ファイルの詳細情報）。画面全体に重ねるダイアログ
//   中身は MediaProbe（player.mediaInfo）が持つ [{title, rows:[{k, v}]}]
Item {
    id: root
    signal closeRequested()
    readonly property var info: player.mediaInfo

    component PlainButton: AbstractButton {
        id: pb
        property bool primary: false
        implicitHeight: 36
        implicitWidth: Math.max(100, pbText.implicitWidth + 32)
        hoverEnabled: true
        focusPolicy: Qt.NoFocus
        opacity: enabled ? 1.0 : 0.4
        background: Rectangle {
            radius: 9
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
            font.family: Theme.font; font.pixelSize: 13
            font.weight: pb.primary ? Font.Bold : Font.Medium
        }
    }

    // ---- 暗幕（後ろの再生画面は触れない）----
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.6)
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; onWheel: (w) => w.accepted = true }
    }

    Keys.onEscapePressed: root.closeRequested()

    Rectangle {
        id: sheet
        anchors.centerIn: parent
        width: Math.min(parent.width - 48, 820)
        height: Math.min(parent.height - 48, 60 + 64 + Math.max(120, body.implicitHeight + 32))
        radius: 14
        color: Theme.bg
        border.color: Theme.border
        clip: true

        ColumnLayout {
            anchors.fill: parent
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
                        paths: ["M21 12a9 9 0 1 1-18 0a9 9 0 1 1 18 0z", "M12 11v6M12 7.5v.5"]
                    }
                    Text { text: "プロパティ"; color: Theme.text; font.family: Theme.font; font.pixelSize: 18; font.bold: true }
                    Text {
                        Layout.fillWidth: true
                        text: root.info ? root.info.fileName : ""
                        elide: Text.ElideMiddle
                        color: Theme.muted; font.family: Theme.font; font.pixelSize: 13
                    }
                    IconButton {
                        implicitWidth: 40; implicitHeight: 40
                        iconSize: 18
                        paths: ["M6 6l12 12M18 6L6 18"]
                        tip: "閉じる（Esc）"
                        onClicked: root.closeRequested()
                    }
                }
            }

            // 本体（セクションごとの表。スクロール）
            Flickable {
                id: flick
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: body.implicitHeight + 32
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: flick.contentHeight > flick.height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff }

                Column {
                    id: body
                    x: 24; y: 16
                    width: flick.width - 48
                    spacing: 18

                    Text {
                        visible: root.info && root.info.busy
                        text: "読み込み中…"
                        color: Theme.muted; font.family: Theme.font; font.pixelSize: 13
                    }
                    Text {
                        visible: root.info && !root.info.busy && root.info.error.length > 0
                        text: root.info ? root.info.error : ""
                        color: Theme.marker; font.family: Theme.font; font.pixelSize: 13
                    }

                    Repeater {
                        model: root.info && !root.info.busy ? root.info.sections : []
                        delegate: Column {
                            id: sec
                            required property var modelData
                            width: body.width
                            spacing: 2

                            Text {
                                text: sec.modelData.title
                                color: Theme.text; font.family: Theme.font; font.pixelSize: 14; font.bold: true
                                bottomPadding: 6
                            }
                            Rectangle { width: parent.width; height: 1; color: Theme.borderSoft }

                            Repeater {
                                model: sec.modelData.rows
                                delegate: Row {
                                    required property var modelData
                                    required property int index
                                    width: sec.width
                                    spacing: 16
                                    Text {
                                        width: 200
                                        topPadding: 4; bottomPadding: 4
                                        text: modelData.k
                                        color: Theme.muted; font.family: Theme.font; font.pixelSize: 13
                                        elide: Text.ElideRight
                                    }
                                    // 値は選択してコピーできる
                                    TextEdit {
                                        width: parent.width - 216
                                        topPadding: 4; bottomPadding: 4
                                        text: modelData.v
                                        readOnly: true
                                        selectByMouse: true
                                        wrapMode: TextEdit.Wrap
                                        color: Theme.text2
                                        selectionColor: Theme.accent
                                        font.family: Theme.font; font.pixelSize: 13
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // フッター
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 64
                color: Theme.bar
                Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Theme.borderSoft }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 10
                    PlainButton { text: "名前を付けて保存…"; enabled: root.info && !root.info.busy && root.info.sections.length > 0; onClicked: root.info.saveAs() }
                    PlainButton { text: "クリップボードにコピー"; enabled: root.info && !root.info.busy && root.info.sections.length > 0; onClicked: root.info.copyToClipboard() }
                    Item { Layout.fillWidth: true }
                    PlainButton { primary: true; text: "閉じる"; onClicked: root.closeRequested() }
                }
            }
        }
    }
}
