import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 下部の操作パネル（時間表示・レンジスライダー・操作行）
Rectangle {
    id: root
    radius: Theme.radius
    color: Theme.panel
    border.color: Theme.borderSoft
    implicitHeight: col.implicitHeight + 26

    readonly property bool active: player.fileOpen && player.controlsEnabled && !player.encoding

    // アイコン（24x24 の SVG パス）
    readonly property var icCutStart: ["M10 4H6v16h4", "M13 12h6"]
    readonly property var icToStart:  ["M6 5v14", "M18 5l-9 7 9 7z"]
    readonly property var icBack:     ["M15 5l-9 7 9 7z", "M19 5v14"]
    readonly property var icReverse:  ["M17 5L7 12l10 7z"]
    readonly property var icPause:    ["M6 5h4v14H6z", "M14 5h4v14h-4z"]
    readonly property var icPlay:     ["M7 4.5l12.5 7.5L7 19.5z"]
    readonly property var icStop:     ["M6 6h12v12H6z"]
    readonly property var icFwd:      ["M5 5v14", "M9 5l9 7-9 7z"]
    readonly property var icToEnd:    ["M6 5l9 7-9 7z", "M18 5v14"]
    readonly property var icCutEnd:   ["M14 4h4v16h-4", "M5 12h6"]

    // スピンボックスの − / ＋
    component SpinButton: AbstractButton {
        id: sb
        property var paths: []
        width: 32
        height: parent.height
        autoRepeat: true
        autoRepeatDelay: 350
        autoRepeatInterval: 60
        focusPolicy: Qt.NoFocus
        hoverEnabled: true
        background: Rectangle { color: sb.down ? Theme.selected : (sb.hovered && sb.enabled ? Theme.hover : "transparent") }
        contentItem: Item {
            Icon {
                anchors.centerIn: parent
                size: 14
                paths: sb.paths
                color: sb.enabled ? Theme.text2 : Theme.faint
            }
        }
    }

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: 14
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        anchors.bottomMargin: 12
        spacing: 10

        // ---- 時間表示 ----
        RowLayout {
            Layout.fillWidth: true
            ColumnLayout {
                spacing: 2
                Layout.minimumWidth: 150
                Text { text: "開始"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 11 }
                Text {
                    textFormat: Text.StyledText
                    text: player.startTime + ' <font color="' + Theme.muted + '">(' + player.rangeStart + ' Frames)</font>'
                    color: Theme.marker; font.family: Theme.mono; font.pixelSize: 13
                }
            }
            Item { Layout.fillWidth: true }
            ColumnLayout {
                spacing: 2
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: player.playTime
                    color: Theme.text; font.family: Theme.mono; font.pixelSize: 24; font.weight: Font.Medium
                }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "再生 · " + player.frameNo + " Frames　/　再生範囲 " + player.rangeTime + " (" + player.rangeFrames + " Frames)"
                    color: Theme.muted; font.family: Theme.font; font.pixelSize: 12
                }
            }
            Item { Layout.fillWidth: true }
            ColumnLayout {
                spacing: 2
                Layout.minimumWidth: 150
                Text { Layout.alignment: Qt.AlignRight; text: "終了"; color: Theme.muted; font.family: Theme.font; font.pixelSize: 11 }
                Text {
                    Layout.alignment: Qt.AlignRight
                    textFormat: Text.StyledText
                    text: player.endTime + ' <font color="' + Theme.muted + '">(' + player.rangeEnd + ' Frames)</font>'
                    color: Theme.marker; font.family: Theme.mono; font.pixelSize: 13
                }
            }
        }

        // ---- レンジスライダー ----
        RangeBar {
            Layout.fillWidth: true
            interactive: root.active
            maximum: Math.max(1, player.maxFrames)
            value: player.frameNo
            rangeStart: player.rangeStart
            rangeEnd: player.rangeEnd
            onScrubbed: (v) => player.scrub(v)
            onReleased: (v) => player.scrubRelease(v)
            onStartMoved: (v) => player.setRangeStartFrame(v)
            onEndMoved: (v) => player.setRangeEndFrame(v)
            onRangeReleased: player.rangeDragFinished()
        }

        // ---- 操作行 ----
        //   再生ボタン群は常に中央に固定する。左右のグループの幅（ジャンプ単位の文字数など）が
        //   変わっても動かないよう、Layout の幅配分ではなくアンカーで置く
        Item {
            Layout.fillWidth: true
            implicitHeight: transportBox.implicitHeight

            // 左: ジャンプ
            RowLayout {
                id: leftGroup
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                spacing: 8
                enabled: root.active

                Rectangle {
                    implicitWidth: seg.implicitWidth + 6
                    implicitHeight: 36
                    radius: 9
                    color: Theme.panel2
                    border.color: Theme.border
                    opacity: parent.enabled ? 1 : 0.4
                    Row {
                        id: seg
                        anchors.centerIn: parent
                        spacing: 2
                        Repeater {
                            model: ["秒", "フレーム", "指定フレーム"]
                            delegate: AbstractButton {
                                required property string modelData
                                required property int index
                                readonly property bool sel: player.jumpMode === index
                                implicitHeight: 30
                                implicitWidth: lbl.implicitWidth + 20
                                focusPolicy: Qt.NoFocus
                                hoverEnabled: true
                                onClicked: player.jumpMode = index
                                background: Rectangle { radius: 6; color: sel ? Theme.selected : (parent.hovered ? Theme.hover : "transparent") }
                                contentItem: Text {
                                    id: lbl
                                    text: modelData
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    color: sel ? "#FFFFFF" : Theme.muted
                                    font.family: Theme.font; font.pixelSize: 12; font.weight: Font.Medium
                                }
                            }
                        }
                    }
                }

                // ジャンプ量（スピンボックス）
                //   ・− / ＋ ボタン（押しっぱなしで連続）
                //   ・ホイール、↑↓ キー（Shift で 10 ずつ）
                //   ・数値は直接入力も可（Enter / フォーカスを外すと確定）
                Rectangle {
                    id: spin
                    readonly property int minValue: player.jumpMode === 2 ? 0 : 1
                    function stepBy(n) {
                        player.jumpValue = Math.max(minValue, player.jumpValue + n)
                        jumpField.text = Qt.binding(() => player.jumpValue)
                    }

                    implicitWidth: 176
                    implicitHeight: 36
                    radius: 8
                    color: Theme.panel2
                    border.color: jumpField.activeFocus ? Theme.accent : Theme.border
                    opacity: parent.enabled ? 1 : 0.4
                    clip: true

                    Row {
                        anchors.fill: parent
                        anchors.margins: 1

                        SpinButton {
                            paths: ["M6 12h12"]
                            enabled: player.jumpValue > spin.minValue
                            onClicked: spin.stepBy(-1)
                        }
                        Rectangle { width: 1; height: parent.height; color: Theme.border }

                        // 値 + 単位
                        Item {
                            width: parent.width - 2 * 32 - 2
                            height: parent.height
                            Row {
                                anchors.centerIn: parent
                                spacing: 4
                                TextInput {
                                    id: jumpField
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: Math.max(24, contentWidth)
                                    horizontalAlignment: TextInput.AlignRight
                                    text: player.jumpValue
                                    validator: IntValidator { bottom: 0; top: 99999999 }
                                    color: Theme.text
                                    selectionColor: Theme.accent
                                    selectByMouse: true
                                    font.family: Theme.mono; font.pixelSize: 13
                                    onEditingFinished: { player.jumpValue = parseInt(text || "0"); text = Qt.binding(() => player.jumpValue) }
                                    Keys.onEscapePressed: { text = Qt.binding(() => player.jumpValue); focus = false }
                                    Keys.onUpPressed:   (e) => spin.stepBy((e.modifiers & Qt.ShiftModifier) ? 10 : 1)
                                    Keys.onDownPressed: (e) => spin.stepBy((e.modifiers & Qt.ShiftModifier) ? -10 : -1)
                                }
                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: player.jumpMode === 0 ? "秒" : (player.jumpMode === 1 ? "フレーム" : "番")
                                    color: Theme.muted; font.family: Theme.font; font.pixelSize: 12
                                }
                            }
                            // ホイールで増減（クリックは TextInput に通す）
                            WheelHandler {
                                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
                                onWheel: (ev) => {
                                    const n = (ev.modifiers & Qt.ShiftModifier) ? 10 : 1
                                    const d = ev.angleDelta.y !== 0 ? ev.angleDelta.y : ev.angleDelta.x
                                    if (d > 0) spin.stepBy(n)
                                    else if (d < 0) spin.stepBy(-n)
                                }
                            }
                        }

                        Rectangle { width: 1; height: parent.height; color: Theme.border }
                        SpinButton {
                            paths: ["M6 12h12", "M12 6v12"]
                            onClicked: spin.stepBy(1)
                        }
                    }
                }
            }

            // 中央: 再生ボタン群
            Rectangle {
                id: transportBox
                // 基本は中央固定。狭くて左のグループに重なるときだけ右へ逃がす
                x: Math.max(leftGroup.width + 12, (parent.width - width) / 2)
                anchors.verticalCenter: parent.verticalCenter
                implicitWidth: transport.implicitWidth + 8
                implicitHeight: transport.implicitHeight + 8
                radius: 14
                color: Theme.panel2
                border.color: Theme.border

                Row {
                    id: transport
                    anchors.centerIn: parent
                    spacing: 4
                    enabled: root.active

                    IconButton { paths: root.icCutStart; iconColor: Theme.marker; strokeWidth: 2.2; tip: "開始位置を設定  ["; onClicked: player.cutStart() }
                    IconButton { paths: root.icToStart; tip: "範囲の先頭へ  |◀"; onClicked: player.toRangeStart() }
                    IconButton {
                        paths: root.icBack
                        caption: (player.jumpMode, player.jumpValue, player.jumpLabel(false))
                        tip: "ジャンプ量だけ戻る  ◀|"
                        onClicked: player.stepBack()
                    }
                    IconButton { paths: root.icReverse; tip: "逆再生  ◀"; onClicked: player.reverse() }

                    // 再生 / 一時停止
                    AbstractButton {
                        id: playBtn
                        implicitWidth: 56; implicitHeight: 48
                        anchors.verticalCenter: parent.verticalCenter
                        focusPolicy: Qt.NoFocus
                        hoverEnabled: true
                        opacity: enabled ? 1 : 0.4
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: "再生 / 一時停止（Space）"
                        onClicked: player.togglePlay()
                        background: Rectangle {
                            radius: 12
                            color: playBtn.down ? Qt.darker(Theme.accent, 1.1) : (playBtn.hovered ? Qt.lighter(Theme.accent, 1.08) : Theme.accent)
                        }
                        contentItem: Item {
                            Icon {
                                anchors.centerIn: parent
                                size: 22
                                filled: true
                                color: Theme.accentInk
                                paths: player.playing && !player.reversing ? root.icPause : root.icPlay
                            }
                        }
                    }

                    IconButton { paths: root.icStop; iconSize: 16; strokeWidth: 2.2; tip: "停止  ■"; onClicked: player.stop() }
                    IconButton {
                        paths: root.icFwd
                        caption: (player.jumpMode, player.jumpValue, player.jumpLabel(true))
                        tip: "ジャンプ量だけ進む  |▶"
                        onClicked: player.stepForward()
                    }
                    IconButton { paths: root.icToEnd; tip: "範囲の末尾へ  ▶|"; onClicked: player.toRangeEnd() }
                    IconButton { paths: root.icCutEnd; iconColor: Theme.marker; strokeWidth: 2.2; tip: "終了位置を設定  ]"; onClicked: player.cutEnd() }
                }
            }

            // 右: 速度・音量・スクリーンショット
            RowLayout {
                id: rightGroup
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                spacing: 8

                ChipButton {
                    id: speedChip
                    enabled: root.active
                    mono: true
                    text: player.speed + "x"
                    paths: ["M4 16a8 8 0 1 1 16 0", "M12 16l4-5"]
                    tip: "再生速度"
                    onClicked: speedPopup.opened ? speedPopup.close() : speedPopup.open()

                    Popup {
                        id: speedPopup
                        y: -height - 8
                        x: (parent.width - width) / 2
                        padding: 6
                        background: Rectangle { radius: 10; color: Theme.panel2; border.color: Theme.border }
                        contentItem: Column {
                            spacing: 2
                            Repeater {
                                model: [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
                                delegate: AbstractButton {
                                    required property real modelData
                                    readonly property bool sel: Math.abs(player.speed - modelData) < 0.0001
                                    width: 88; height: 32
                                    hoverEnabled: true
                                    focusPolicy: Qt.NoFocus
                                    onClicked: { player.speed = modelData; speedPopup.close() }
                                    background: Rectangle { radius: 6; color: sel ? Theme.selected : (parent.hovered ? Theme.hover : "transparent") }
                                    contentItem: Text {
                                        text: modelData + "x"
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                        color: sel ? Theme.accent : Theme.text
                                        font.family: Theme.mono; font.pixelSize: 13
                                    }
                                }
                            }
                        }
                    }
                }

                ChipButton {
                    id: volChip
                    mono: true
                    text: player.volume
                    paths: player.volume === 0 ? ["M4 9v6h4l5 4V5L8 9z", "M17 9l4 6M21 9l-4 6"]
                                               : ["M4 9v6h4l5 4V5L8 9z", "M16.5 9a4 4 0 0 1 0 6"]
                    tip: "音量"
                    onClicked: volPopup.opened ? volPopup.close() : volPopup.open()

                    Popup {
                        id: volPopup
                        y: -height - 8
                        x: (parent.width - width) / 2
                        padding: 12
                        background: Rectangle { radius: 10; color: Theme.panel2; border.color: Theme.border }
                        contentItem: Column {
                            spacing: 6
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: player.volume
                                color: Theme.text; font.family: Theme.mono; font.pixelSize: 12
                            }
                            Slider {
                                id: volSlider
                                orientation: Qt.Vertical
                                height: 140
                                from: 0; to: 100; stepSize: 1
                                value: player.volume
                                onMoved: player.volume = Math.round(value)
                                background: Rectangle {
                                    x: volSlider.leftPadding + volSlider.availableWidth / 2 - 2
                                    y: volSlider.topPadding
                                    width: 4; height: volSlider.availableHeight; radius: 2
                                    color: Theme.border
                                    Rectangle {
                                        width: 4; radius: 2
                                        y: volSlider.visualPosition * parent.height
                                        height: parent.height - y
                                        color: Theme.accent
                                    }
                                }
                                handle: Rectangle {
                                    x: volSlider.leftPadding + volSlider.availableWidth / 2 - 8
                                    y: volSlider.topPadding + volSlider.visualPosition * (volSlider.availableHeight - 16)
                                    width: 16; height: 16; radius: 8
                                    color: "#FFFFFF"
                                }
                            }
                        }
                    }
                }

                // スクリーンショット（保存 ＋ 保存先）
                Rectangle {
                    implicitWidth: shotRow.implicitWidth + 2
                    implicitHeight: 36
                    radius: 8
                    color: Theme.panel2
                    border.color: Theme.border
                    clip: true
                    Row {
                        id: shotRow
                        anchors.centerIn: parent
                        AbstractButton {
                            id: shotBtn
                            enabled: root.active
                            implicitHeight: 34
                            implicitWidth: shotInner.implicitWidth + 24
                            hoverEnabled: true
                            focusPolicy: Qt.NoFocus
                            opacity: enabled ? 1 : 0.4
                            ToolTip.visible: hovered
                            ToolTip.delay: 500
                            ToolTip.text: "スクリーンショットを保存"
                            onClicked: player.screenshot()
                            background: Rectangle { color: shotBtn.hovered ? Theme.hover : "transparent" }
                            contentItem: Item {
                                Row {
                                    id: shotInner
                                    anchors.centerIn: parent
                                    spacing: 8
                                    Icon { anchors.verticalCenter: parent.verticalCenter; size: 16; paths: ["M4 8h3l2-3h6l2 3h3v11H4z", "M15.5 13a3.5 3.5 0 1 1-7 0a3.5 3.5 0 1 1 7 0z"] }
                                    Text { anchors.verticalCenter: parent.verticalCenter; text: "保存"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.weight: Font.Medium }
                                }
                            }
                        }
                        Rectangle { width: 1; height: 34; color: Theme.border }
                        IconButton {
                            implicitWidth: 36; implicitHeight: 34
                            iconSize: 16
                            enabled: player.ready
                            paths: ["M3 7h6l2 2h10v10H3z"]
                            tip: "スクリーンショットの保存先・形式"
                            onClicked: player.openScreenshotSettings()
                        }
                    }
                }
            }
        }
    }
}
