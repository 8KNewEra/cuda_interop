import QtQuick
import QtQuick.Layouts

// 動画情報オーバーレイ（旧: paintGL の QPainter テキスト）
Rectangle {
    id: root
    property var info: ({})
    property var gpus: []
    // ★ フレーム番号は毎フレーム変わるので info（500ms 更新）とは別に受け取る
    property int frameNo: 0
    property real speed: 1.0
    property bool playing: false

    // 描画性能（実測 / 目標）。目標 = 動画 fps × 再生速度
    readonly property real targetFps: (info.fps || 0) * speed
    readonly property real renderFps: info.renderFps || 0
    readonly property real renderRatio: targetFps > 0 ? renderFps / targetFps : 0
    readonly property color renderColor: !playing ? Theme.muted2
                                       : renderRatio >= 0.95 ? Theme.accent
                                       : renderRatio >= 0.75 ? Theme.marker : "#FF7A7A"

    width: 360
    implicitHeight: col.implicitHeight + 28
    radius: Theme.radius
    color: Theme.overlay
    border.color: Theme.border

    component Key: Text { color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.preferredWidth: 112 }
    component Val: Text { color: Theme.text2; font.family: Theme.font; font.pixelSize: 12; wrapMode: Text.WrapAnywhere; maximumLineCount: 2; elide: Text.ElideRight; Layout.fillWidth: true }
    component Num: Text { color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12; Layout.fillWidth: true }
    component Meter: RowLayout {
        property string label
        property real   ratio: 0
        property string valueText
        property color  barColor: Theme.accent
        Layout.fillWidth: true
        spacing: 8
        Text { text: label; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.preferredWidth: 30 }
        Rectangle {
            Layout.fillWidth: true; height: 4; radius: 2; color: Theme.border
            Rectangle { width: parent.width * Math.max(0, Math.min(1, ratio)); height: 4; radius: 2; color: barColor }
        }
        Text { text: valueText; color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: 96 }
    }

    function fmtFps(v) { return (v === undefined || v === null) ? "-" : Number(v).toFixed(2) }

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: 14
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Text { text: "動画情報"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.bold: true }
            Item { Layout.fillWidth: true }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 12
            rowSpacing: 3

            Key { text: "File Name" }        Val { text: root.info.fileName || "-" }
            Key { text: "Decoder" }          Val { text: root.info.decoder || "-" }
            Key { text: "Decode Mode" }      Val { text: root.info.decodeMode || "-" }
            Key { text: "Resolution" }       Num { text: (root.info.width || 0) + "×" + (root.info.height || 0) }
            Key { text: "Video Framerate" }  Num { text: root.fmtFps(root.info.fps) }
            Key { text: "Max Frame" }        Num { text: root.info.maxFrame !== undefined ? root.info.maxFrame : "-" }
            Key { text: "FrameNo" }          Num { text: root.frameNo }
            Key { text: "Audio Channels"; visible: !!root.info.hasAudio }
            Num { text: root.info.audioChannels !== undefined ? root.info.audioChannels : "-"; visible: !!root.info.hasAudio }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

        Repeater {
            model: root.gpus
            delegate: ColumnLayout {
                required property var modelData
                Layout.fillWidth: true
                spacing: 3

                // 1 行目: GPU 名（省略しない）、2 行目: デバイス番号と役割
                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    text: modelData.name
                    color: Theme.text2; font.family: Theme.font; font.pixelSize: 12; font.weight: Font.Medium
                }
                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        Layout.fillWidth: true
                        text: "CUDA " + modelData.id + (modelData.primary ? " · Primary (OpenGL)" : "")
                        color: modelData.primary ? Theme.accent : Theme.muted2
                        font.family: Theme.font; font.pixelSize: 11
                    }
                    Text { text: modelData.sm; color: Theme.muted2; font.family: Theme.mono; font.pixelSize: 11 }
                }

                Meter { label: "GPU"; ratio: modelData.usage / 100; valueText: modelData.usage + "%" }
                Meter {
                    label: "MEM"; barColor: Theme.muted2
                    ratio: modelData.memMax > 0 ? modelData.memUsed / modelData.memMax : 0
                    valueText: modelData.memUsed + "/" + modelData.memMax + " MB"
                }
            }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

        // ★ 描画フレームレート：OpenGL で実際に表示できた枚数を、目標と並べて出す
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            RowLayout {
                Layout.fillWidth: true
                Text { text: "描画フレームレート"; color: Theme.text2; font.family: Theme.font; font.pixelSize: 12; font.weight: Font.Medium }
                Item { Layout.fillWidth: true }
                Text {
                    textFormat: Text.StyledText
                    text: root.playing
                          ? '<font color="' + root.renderColor + '">' + Number(root.renderFps).toFixed(1) + '</font>'
                            + ' / ' + Number(root.targetFps).toFixed(2) + ' fps'
                          : '一時停止中'
                    color: Theme.muted2; font.family: Theme.mono; font.pixelSize: 12
                }
            }
            Rectangle {
                Layout.fillWidth: true; height: 4; radius: 2; color: Theme.border
                Rectangle {
                    width: parent.width * Math.max(0, Math.min(1, root.playing ? root.renderRatio : 0))
                    height: 4; radius: 2; color: root.renderColor
                }
            }
            Text {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                text: "実測 / 目標（動画 " + root.fmtFps(root.info.fps) + " fps × 再生速度 " + root.speed + "x）"
                color: Theme.faint; font.family: Theme.font; font.pixelSize: 11
            }
        }
    }
}
