import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import CudaViewer

// 通常再生画面（Design キャンバスのモック準拠）
ApplicationWindow {
    id: win
    width: 1440
    height: 900
    minimumWidth: 1200      // 操作行（ジャンプ・再生ボタン・速度など）が重ならない幅
    minimumHeight: 600
    visible: true
    title: "CUDA Master VideoViewer"
    color: Theme.bg

    property bool panelOpen: false      // 画像処理パネルは起動時は閉じておく
    property bool fullscreenVideo: false   // 映像だけを全画面（旧: 最大化で全画面）

    font.family: Theme.font

    // タイトルバーのウィンドウ操作ボタン
    component WinButton: AbstractButton {
        id: wb
        property var  paths: []
        property bool danger: false
        width: 46
        height: parent.height
        hoverEnabled: true
        focusPolicy: Qt.NoFocus
        background: Rectangle {
            color: wb.danger ? (wb.down ? "#A52A1E" : (wb.hovered ? "#C42B1C" : "transparent"))
                             : (wb.down ? Theme.selected : (wb.hovered ? Theme.hover : "transparent"))
        }
        contentItem: Item {
            Icon {
                anchors.centerIn: parent
                size: 16
                strokeWidth: 1.4
                paths: wb.paths
                color: wb.danger && wb.hovered ? "#FFFFFF" : Theme.text2
            }
        }
    }

    function toggleFullscreenVideo() {
        fullscreenVideo = !fullscreenVideo
        if (fullscreenVideo) win.showFullScreen()
        else win.showNormal()
    }

    // ★ エンコード設定を開いている間は再生系のショートカットを止める
    Shortcut { sequence: "Space"; context: Qt.ApplicationShortcut; enabled: !win.encodeOpen; onActivated: player.togglePlay() }
    Shortcut { sequence: "F11";   context: Qt.ApplicationShortcut; enabled: !win.encodeOpen; onActivated: win.toggleFullscreenVideo() }
    Shortcut {
        sequence: "Esc"; context: Qt.ApplicationShortcut
        enabled: win.fullscreenVideo
        onActivated: win.toggleFullscreenVideo()
    }
    Shortcut { sequences: [StandardKey.Open]; enabled: !win.encodeOpen; onActivated: player.openFileDialog() }

    // ---- エンコード設定（再生画面の上に重ねる）----
    property bool encodeOpen: false
    Connections { target: player;  function onEncodeSettingsOpened() { win.encodeOpen = true } }
    Connections { target: encoder; function onClosed() { win.encodeOpen = false } }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ===== ヘッダー（= タイトルバー）=====
        //   ★ OS のタイトルバーは WindowChrome が消している。このヘッダーの
        //     ボタン以外の部分をドラッグするとウィンドウが動き、
        //     ダブルクリックで最大化／元に戻す（Windows 標準の動作）。
        Rectangle {
            id: titleBar
            visible: !win.fullscreenVideo
            Layout.fillWidth: true
            implicitHeight: 48
            color: Theme.bar
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.borderSoft }

            Component.onCompleted: if (chrome.active) chrome.titleBar = titleBar

            // ウィンドウ操作ボタン（最小化・最大化・閉じる）
            Row {
                id: windowButtons
                visible: chrome.active
                anchors.right: parent.right
                anchors.top: parent.top
                height: parent.height - 1

                WinButton {
                    paths: ["M6 12h12"]
                    ToolTip.visible: hovered; ToolTip.delay: 800; ToolTip.text: "最小化"
                    onClicked: win.showMinimized()
                }
                WinButton {
                    readonly property bool maxed: win.visibility === Window.Maximized
                    paths: maxed ? ["M8.5 8.5h9v9h-9z", "M6.5 15.5v-9h9"] : ["M6.5 6.5h11v11h-11z"]
                    ToolTip.visible: hovered; ToolTip.delay: 800; ToolTip.text: maxed ? "元に戻す" : "最大化"
                    onClicked: maxed ? win.showNormal() : win.showMaximized()
                }
                WinButton {
                    danger: true
                    paths: ["M6.5 6.5l11 11M17.5 6.5l-11 11"]
                    ToolTip.visible: hovered; ToolTip.delay: 800; ToolTip.text: "閉じる"
                    onClicked: win.close()
                }
            }

            RowLayout {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.right: windowButtons.visible ? windowButtons.left : parent.right
                // ★ エンコード設定中はヘッダーの操作も止める（ウィンドウ操作ボタンとドラッグは残す）
                enabled: !win.encodeOpen
                anchors.leftMargin: 16
                anchors.rightMargin: windowButtons.visible ? 12 : 16
                spacing: 16

                Row {
                    spacing: 10
                    Icon {
                        anchors.verticalCenter: parent.verticalCenter
                        size: 22; color: Theme.accent
                        paths: ["M6 5h12a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3z", "M10 9.5v5l4.5-2.5z"]
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "CUDA Master VideoViewer"
                        color: Theme.text; font.pixelSize: 14; font.bold: true; font.letterSpacing: 0.3
                    }
                }

                // ファイルメニュー
                AbstractButton {
                    id: fileMenuBtn
                    implicitHeight: 32
                    implicitWidth: fileLbl.implicitWidth + 24
                    hoverEnabled: true
                    focusPolicy: Qt.NoFocus
                    onClicked: fileMenu.opened ? fileMenu.close() : fileMenu.open()
                    background: Rectangle { radius: 6; color: fileMenuBtn.hovered || fileMenu.opened ? Theme.hover : "transparent" }
                    contentItem: Text {
                        id: fileLbl
                        text: "ファイル"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        color: Theme.text2; font.pixelSize: 13; font.weight: Font.Medium
                    }
                    AppMenu {
                        id: fileMenu
                        y: fileMenuBtn.height + 4
                        // hint = 右端に出すキー表示（実際のキーは上の Shortcut で処理）
                        Action { property string hint: "Ctrl+O"; text: "ファイルを開く…"; enabled: player.ready; onTriggered: player.openFileDialog() }
                        Action {
                            text: "ファイルを保存…"
                            enabled: player.fileOpen && player.controlsEnabled && !player.encoding
                            onTriggered: player.openEncodeSettings()
                        }
                        Action { text: "ファイルを閉じる"; enabled: player.fileOpen; onTriggered: player.closeFile() }
                        AppMenuSeparator {}
                        Action { property string hint: "F11"; text: "全画面"; onTriggered: win.toggleFullscreenVideo() }
                    }
                }

                // 表示トグル
                Rectangle {
                    implicitWidth: dispRow.implicitWidth + 6
                    implicitHeight: 34
                    radius: 9
                    color: Theme.panel2
                    border.color: Theme.border
                    Row {
                        id: dispRow
                        anchors.centerIn: parent
                        spacing: 2
                        Repeater {
                            model: [
                                { label: "動画情報",     key: "infoVisible", paths: ["M21 12a9 9 0 1 1-18 0a9 9 0 1 1 18 0z", "M12 11v6M12 7.5v.5"] },
                                { label: "ヒストグラム", key: "histVisible", paths: ["M4 20h16M7 20v-6M11 20V8M15 20v-9M19 20v-4"] }
                            ]
                            delegate: AbstractButton {
                                id: dispBtn
                                required property var modelData
                                // ★ ファイルを開いていない間は押せない（表示も OFF に見せる）
                                readonly property bool on: player[modelData.key] && enabled
                                enabled: player.fileOpen
                                implicitHeight: 28
                                implicitWidth: dispInner.implicitWidth + 20
                                hoverEnabled: true
                                focusPolicy: Qt.NoFocus
                                onClicked: player[modelData.key] = !player[modelData.key]
                                opacity: enabled ? 1.0 : 0.4
                                background: Rectangle { radius: 6; color: dispBtn.on ? Theme.selected : (dispBtn.hovered ? Theme.hover : "transparent") }
                                contentItem: Item {
                                    Row {
                                        id: dispInner
                                        anchors.centerIn: parent
                                        spacing: 6
                                        Icon { anchors.verticalCenter: parent.verticalCenter; size: 15; paths: dispBtn.modelData.paths; color: dispBtn.on ? "#FFFFFF" : Theme.muted2 }
                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: dispBtn.modelData.label
                                            color: dispBtn.on ? "#FFFFFF" : Theme.muted2
                                            font.pixelSize: 12; font.weight: Font.Medium
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    horizontalAlignment: Text.AlignHCenter
                    elide: Text.ElideMiddle
                    text: player.fileOpen ? (player.videoInfo.fileName || "") : "ファイルをドロップするか、ファイル › ファイルを開く"
                    color: Theme.muted; font.pixelSize: 13
                }

                ChipButton {
                    text: "画像処理"
                    paths: ["M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12", "M18 6a2 2 0 1 1-4 0a2 2 0 1 1 4 0z", "M12 12a2 2 0 1 1-4 0a2 2 0 1 1 4 0z"]
                    badge: (player.gaussian ? 1 : 0) + (player.sobel ? 1 : 0) + (player.averaging ? 1 : 0)
                    checkedLook: win.panelOpen && player.fileOpen
                    enabled: player.fileOpen
                    tip: "画像処理パネルの開閉"
                    onClicked: win.panelOpen = !win.panelOpen
                }

                Rectangle { width: 1; height: 20; color: Theme.border }

                ChipButton {
                    primary: true
                    text: "ファイルを保存"
                    paths: ["M12 4v11", "M7 10l5 5 5-5", "M5 20h14"]
                    enabled: player.fileOpen && player.controlsEnabled && !player.encoding
                    tip: "エンコード設定を開く"
                    onClicked: player.openEncodeSettings()
                }
            }
        }

        // ===== 本体（映像 + 操作パネル）=====
        //   ★ エンコード設定を開いている間は、開いた瞬間の見た目を 1 枚の
        //     テクスチャに焼いて表示し、本体は描かない（hideSource）。
        //     後ろでスライダーや映像が動かず、追加の描画も起きない。
        Item {
            id: bodyHost
            Layout.fillWidth: true
            Layout.fillHeight: true

        ColumnLayout {
            id: body
            anchors.fill: parent
            spacing: 0

        // ===== 映像 + 画像処理パネル =====
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: win.fullscreenVideo ? 0 : 12
            Layout.bottomMargin: 0
            spacing: 12

            Rectangle {
                id: videoFrame
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumHeight: 240
                radius: win.fullscreenVideo ? 0 : Theme.radius
                color: "#000000"
                border.color: win.fullscreenVideo ? "transparent" : Theme.borderSoft
                clip: true

                VideoItem {
                    id: video
                    anchors.fill: parent
                    anchors.margins: win.fullscreenVideo ? 0 : 1
                    interactive: player.fileOpen && !player.encoding
                }

                // 未オープン時の案内
                Column {
                    anchors.centerIn: parent
                    spacing: 8
                    visible: !video.hasVideo
                    Icon {
                        anchors.horizontalCenter: parent.horizontalCenter
                        size: 40; strokeWidth: 1.5; color: "#5F6670"
                        paths: ["M5 5h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z", "M3 9h18M8 5v4M16 5v4"]
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: player.fileOpen ? "読み込み中…" : (player.ready ? "動画ファイルをここへドロップ" : "GPU を初期化しています…")
                        color: "#8A919B"; font.pixelSize: 15; font.weight: Font.Medium
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: "mp4 / mkv / mov / avi"
                        color: Theme.faint; font.pixelSize: 12
                    }
                }

                DropArea {
                    anchors.fill: parent
                    onEntered: (drag) => { drag.accepted = drag.hasUrls && player.isSupportedFile(drag.urls[0]) }
                    onDropped: (drop) => { if (drop.hasUrls) player.openFile(drop.urls[0]) }
                }

                InfoPanel {
                    visible: player.infoVisible && player.fileOpen && !win.fullscreenVideo
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.margins: 16
                    width: Math.min(360, parent.width - 32)
                    height: implicitHeight
                    info: player.videoInfo
                    gpus: player.gpus
                    frameNo: player.frameNo
                    speed: player.speed
                    playing: player.renderActive
                }

                ZoomNavigator {
                    id: zoomNav
                    visible: video.zoom > 1.0001
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.margins: 16
                    // ★ 映像枠に比例（大きな画面では大きく）
                    width: Math.max(200, Math.min(360, videoFrame.width * 0.2))
                    zoom: video.zoom
                    view: video.viewRect
                    source: video.sourceSize
                }

                HistogramPanel {
                    visible: player.histVisible && player.fileOpen && !win.fullscreenVideo
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: 16
                    // ★ 映像枠に比例（最小 360、最大 640）
                    // 　 縦に収まらない小さなウィンドウでは高さ側で頭打ちにする
                    width: Math.min(parent.width - 32,
                                    1.5 * (videoFrame.height - 32 - 180 - (zoomNav.visible ? zoomNav.height + 16 : 0)) + 84,
                                    Math.max(360, Math.min(640, videoFrame.width * 0.32)))
                    height: implicitHeight
                    stats: player.histogram
                }

                // エンコード中の表示
                Rectangle {
                    visible: player.encoding
                    anchors.top: parent.top
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.topMargin: 16
                    width: encText.implicitWidth + 28
                    height: 32
                    radius: 16
                    color: Theme.overlay
                    border.color: Theme.marker
                    Text {
                        id: encText
                        anchors.centerIn: parent
                        text: "エンコードモード（設定ウィンドウで操作）"
                        color: Theme.marker; font.pixelSize: 12; font.weight: Font.Medium
                    }
                }
            }

            FilterPanel {
                visible: win.panelOpen && player.fileOpen && !win.fullscreenVideo
                Layout.fillHeight: true
                Layout.preferredWidth: 300
                onCloseRequested: win.panelOpen = false
            }
        }

        // ===== 操作パネル =====
        ControlPanel {
            visible: !win.fullscreenVideo
            Layout.fillWidth: true
            Layout.margins: 12
        }
        }   // body

            // 開くたびに作り直す → その時点の画面を 1 回だけ撮って固定
            Loader {
                anchors.fill: parent
                active: win.encodeOpen
                sourceComponent: ShaderEffectSource {
                    sourceItem: body
                    live: false          // 以後は再描画しない
                    hideSource: true     // 本体は描かない
                    smooth: true
                }
            }
        }   // bodyHost
    }

    EncodePanel {
        // ヘッダー（タイトルバー）は覆わない：移動・最小化・閉じるは使えるように
        anchors.fill: parent
        anchors.topMargin: titleBar.visible ? titleBar.height : 0
        visible: win.encodeOpen
        focus: visible
        onCloseRequested: encoder.close()
    }
}
