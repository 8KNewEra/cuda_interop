import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import CudaViewer

// メインウィンドウ（QML 版 UI のルート）
//   構成: ヘッダー（自前タイトルバー＋メニュー）/ 本体（映像＋画像処理パネル）/ 操作パネル
//   映像の上に重ねる表示: タイル境界・画素グリッド・スポイト・比較表示の境目
//   画面全体を覆うダイアログ: エンコード設定・画像保存設定・プロパティ
//   C++ 側の窓口は context property（player / encoder / shotSettings）
ApplicationWindow {
    id: win
    // スポイトの 1 行（色見本・ラベル・R/G/B/輝度）
    // 注意: inline component はファイル直下に置く（入れ子にすると Qt 6.9 で解決できない）
    component EyeRow: Row {
        property string label
        property int r; property int g; property int b
        spacing: 8
        Rectangle { width: 12; height: 12; radius: 3; anchors.verticalCenter: parent.verticalCenter
                    color: Qt.rgba(r / 255, g / 255, b / 255, 1); border.color: Theme.border }
        Text { text: label; width: 44; color: Theme.muted; font.family: Theme.font; font.pixelSize: 11; anchors.verticalCenter: parent.verticalCenter }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            textFormat: Text.StyledText
            // 桁をそろえるため &nbsp; で詰める（リッチテキストでは通常の空白が縮む）
            function p3(v) { const t = String(v); return "&nbsp;".repeat(Math.max(0, 3 - t.length)) + t }
            text: '<font color="#FF7A7A">R</font>&nbsp;' + p3(r)
                + '&nbsp;&nbsp;<font color="#6EE7A0">G</font>&nbsp;' + p3(g)
                + '&nbsp;&nbsp;<font color="#7AB8FF">B</font>&nbsp;' + p3(b)
                + '&nbsp;&nbsp;<font color="#9BA3AE">Y</font>&nbsp;' + p3(Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b))
            color: Theme.text; font.family: Theme.mono; font.pixelSize: 12
        }
    }

    width: 1440
    height: 900
    minimumWidth: 1200      // 操作行（ジャンプ・再生ボタン・速度など）が重ならない幅
    minimumHeight: 600
    visible: true
    title: "CUDA Master VideoViewer"
    color: Theme.bg

    property bool panelOpen: false      // 画像処理パネルは起動時は閉じておく
    property bool fullscreenVideo: false   // 映像だけを全画面表示

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

    // 再生系のショートカット。ダイアログ表示中・重い処理中は止める（canOperate）
    Shortcut { sequence: "Space"; context: Qt.ApplicationShortcut; enabled: win.canOperate && player.fileOpen; onActivated: player.togglePlay() }
    Shortcut { sequence: "F11";   context: Qt.ApplicationShortcut; enabled: win.canOperate; onActivated: win.toggleFullscreenVideo() }
    Shortcut {
        sequence: "Esc"; context: Qt.ApplicationShortcut
        enabled: win.fullscreenVideo
        onActivated: win.toggleFullscreenVideo()
    }
    Shortcut { sequence: "Ctrl+I"; enabled: win.canOperate && player.fileOpen; onActivated: player.showProperties() }
    Shortcut { sequences: [StandardKey.Open]; enabled: win.canOperate && player.ready; onActivated: player.openFileDialog() }

    // ---- エンコード設定（再生画面の上に重ねる）----
    property bool encodeOpen: false
    Connections { target: player;  function onEncodeSettingsOpened() { win.encodeOpen = true } }
    Connections { target: encoder; function onClosed() { win.encodeOpen = false } }

    // ---- 画像保存設定（同じく重ねて出す）----
    property bool shotOpen: false
    Connections { target: shotSettings; function onOpened() { win.shotOpen = true } function onClosed() { win.shotOpen = false } }

    // ---- プロパティ（ファイルの詳細情報）----
    property bool propsOpen: false
    Connections { target: player.mediaInfo; function onOpened() { win.propsOpen = true } }

    // ダイアログ表示中はヘッダーとショートカットを止める
    readonly property bool modalOpen: encodeOpen || shotOpen || propsOpen
    // 操作可能か（ダイアログなし かつ 重い処理中でない）
    // busy: 高精度シーク・画像保存などの重い処理中。controlsEnabled は未オープン時も
    //   false なので「ファイルを開いていて操作不可」のときだけ busy とする。
    //   busy 中に使えるのは「ファイルを閉じる」とウィンドウ操作だけ。
    readonly property bool busy: player.fileOpen && !player.controlsEnabled
    readonly property bool canOperate: !modalOpen && !busy

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ===== ヘッダー（= タイトルバー）=====
        //   OS のタイトルバーは WindowChrome が消している。ボタン以外をドラッグで移動、
        //   ダブルクリックで最大化／元に戻す。
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
                // ダイアログ表示中はメニュー・トグルを止める（ウィンドウ操作ボタンとドラッグは残す）
                enabled: !win.modalOpen
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
                        // hint = 右端のキー表示（キー操作自体は上の Shortcut で処理）
                        Action { property string hint: "Ctrl+O"; text: "ファイルを開く…"; enabled: player.ready && !win.busy; onTriggered: player.openFileDialog() }
                        Action {
                            text: "ファイルを保存…"
                            enabled: player.fileOpen && player.controlsEnabled && !player.encoding
                            onTriggered: player.openEncodeSettings()
                        }
                        Action { text: "ファイルを閉じる"; enabled: player.fileOpen; onTriggered: player.closeFile() }
                        AppMenuSeparator {}
                        Action { property string hint: "Ctrl+I"; text: "プロパティ…"; enabled: player.fileOpen && !win.busy; onTriggered: player.showProperties() }
                        AppMenuSeparator {}
                        Action { property string hint: "F11"; text: "全画面"; enabled: !win.busy; onTriggered: win.toggleFullscreenVideo() }
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
                                // 未オープン時は押せず OFF 表示。重い処理中は押せないが ON/OFF の見た目は保つ
                                readonly property bool on: player[modelData.key] && player.fileOpen
                                enabled: player.fileOpen && player.controlsEnabled
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
                    text: player.fileOpen ? (player.videoInfo.fileName || "") : "動画ファイルをドロップするか、［ファイル］メニューから開いてください"
                    color: Theme.muted; font.pixelSize: 13
                }

                ChipButton {
                    text: "画像処理"
                    paths: ["M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12", "M18 6a2 2 0 1 1-4 0a2 2 0 1 1 4 0z", "M12 12a2 2 0 1 1-4 0a2 2 0 1 1 4 0z"]
                    badge: (player.imageAdjust ? 1 : 0) + (player.colorCorrect ? 1 : 0) + (player.toneCurve ? 1 : 0) + (player.gaussian ? 1 : 0) + (player.sobel ? 1 : 0) + (player.averaging ? 1 : 0) + (player.compare ? 1 : 0) + (player.channelView ? 1 : 0) + (player.tileSeam ? 1 : 0) + (player.eyedropper ? 1 : 0) + (player.zebra ? 1 : 0)
                    checkedLook: win.panelOpen && player.fileOpen
                    enabled: player.fileOpen && player.controlsEnabled
                    tip: "画像処理パネルの表示 / 非表示"
                    onClicked: win.panelOpen = !win.panelOpen
                }

                Rectangle { width: 1; height: 20; color: Theme.border }

                ChipButton {
                    primary: true
                    text: "ファイルを保存"
                    paths: ["M12 4v11", "M7 10l5 5 5-5", "M5 20h14"]
                    enabled: player.fileOpen && player.controlsEnabled && !player.encoding
                    tip: "エンコード設定を開きます"
                    onClicked: player.openEncodeSettings()
                }
            }
        }

        // ===== 本体（映像 + 操作パネル）=====
        //   エンコード設定を開いている間は、開いた瞬間の画面を 1 枚のテクスチャにして
        //   表示し本体は描かない（hideSource）。後ろの映像やスライダーは止まって見える。
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
                    onEntered: (drag) => { drag.accepted = win.canOperate && drag.hasUrls && player.isSupportedFile(drag.urls[0]) }
                    onDropped: (drop) => { if (drop.hasUrls) player.openFile(drop.urls[0]) }
                }

                InfoPanel {
                    z: 2      // 重なり順: タイル境界(1) < 各パネル(2) < 比較表示の境目(3)
                    visible: player.infoVisible && player.fileOpen && !win.fullscreenVideo
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.margins: 16
                    width: Math.min(360, parent.width - 32)
                    height: Math.min(implicitHeight, parent.height - 32)   // 入りきらなければ中をスクロール
                    info: player.videoInfo
                    gpus: player.gpus
                    fileOpen: player.fileOpen
                    frameNo: player.frameNo
                    speed: player.speed
                    playing: player.renderActive
                }

                ZoomNavigator {
                    z: 2      // 重なり順は上と同じ
                    id: zoomNav
                    visible: video.zoom > 1.0001
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.margins: 16
                    // 映像枠に比例
                    width: Math.max(200, Math.min(360, videoFrame.width * 0.2))
                    zoom: video.zoom
                    view: video.viewRect
                    source: video.sourceSize
                }

                HistogramPanel {
                    z: 2      // 重なり順は上と同じ
                    visible: player.histVisible && player.fileOpen && !win.fullscreenVideo
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: 16
                    // 映像枠に比例（360〜640）。小さいウィンドウでは高さ側で頭打ち
                    width: Math.min(parent.width - 32,
                                    1.5 * (videoFrame.height - 32 - 180 - (zoomNav.visible ? zoomNav.height + 16 : 0)) + 84,
                                    Math.max(360, Math.min(640, videoFrame.width * 0.32)))
                    height: implicitHeight
                    stats: player.histogram
                }

                // ---- タイル境界線とタイル番号（タイル境界チェック）----
                //   ズーム・パンに追従（viewRect から画面座標に変換）
                Item {
                    id: tileGrid
                    z: 1          // 各パネルより下
                    anchors.fill: video
                    clip: true
                    readonly property int cols: player.videoInfo.tileCols || 1
                    readonly property int rows: player.videoInfo.tileRows || 1
                    visible: player.tileSeam && (player.filterParams.seamGrid ?? true) && cols * rows > 1
                             && player.fileOpen && video.hasVideo && !player.encoding
                    readonly property rect vr: video.videoRect
                    readonly property rect sr: video.viewRect
                    readonly property color lineColor: (Theme.seamColors[player.filterParams.seamGridColor ?? 0] || Theme.seamColors[0]).c
                    function sx(f) { return vr.x + (f - sr.x) / Math.max(1e-6, sr.width)  * vr.width }
                    function sy(f) { return vr.y + (f - sr.y) / Math.max(1e-6, sr.height) * vr.height }

                    Repeater {          // 縦線
                        model: tileGrid.visible ? tileGrid.cols - 1 : 0
                        delegate: Rectangle {
                            required property int index
                            x: Math.round(tileGrid.sx((index + 1) / tileGrid.cols)); y: tileGrid.vr.y
                            width: 1; height: tileGrid.vr.height
                            color: tileGrid.lineColor
                            opacity: 0.85
                        }
                    }
                    Repeater {          // 横線
                        model: tileGrid.visible ? tileGrid.rows - 1 : 0
                        delegate: Rectangle {
                            required property int index
                            x: tileGrid.vr.x; y: Math.round(tileGrid.sy((index + 1) / tileGrid.rows))
                            width: tileGrid.vr.width; height: 1
                            color: tileGrid.lineColor
                            opacity: 0.85
                        }
                    }
                    Repeater {          // タイル番号（左上。デコード順＝左上から行ごと）
                        model: tileGrid.visible ? tileGrid.cols * tileGrid.rows : 0
                        delegate: Rectangle {
                            required property int index
                            readonly property real lx: tileGrid.sx((index % tileGrid.cols) / tileGrid.cols)
                            readonly property real ly: tileGrid.sy(Math.floor(index / tileGrid.cols) / tileGrid.rows)
                            x: Math.max(tileGrid.vr.x, lx) + 8
                            y: Math.max(tileGrid.vr.y, ly) + 8
                            // タイルが画面外なら出さない
                            visible: lx < tileGrid.vr.x + tileGrid.vr.width && ly < tileGrid.vr.y + tileGrid.vr.height
                                     && tileGrid.sx(((index % tileGrid.cols) + 1) / tileGrid.cols) > tileGrid.vr.x
                                     && tileGrid.sy((Math.floor(index / tileGrid.cols) + 1) / tileGrid.rows) > tileGrid.vr.y
                            width: tileLbl.implicitWidth + 12; height: 20; radius: 4
                            color: Qt.rgba(0, 0, 0, 0.6)
                            Text { id: tileLbl; anchors.centerIn: parent; text: "Tile " + parent.index; color: tileGrid.lineColor; font.family: Theme.mono; font.pixelSize: 11 }
                        }
                    }
                }

                // ---- 画素グリッド: 1 画素が 12px 以上に拡大されたとき画素の境目を描く ----
                Canvas {
                    id: pixGrid
                    z: 1
                    anchors.fill: video
                    readonly property rect vr: video.videoRect
                    readonly property rect sr: video.viewRect
                    readonly property size src: video.sourceSize
                    // 画面上での 1 画素の大きさ（px）
                    readonly property real pxSize: src.width > 0 ? vr.width / (sr.width * src.width) : 0
                    visible: video.hasVideo && player.fileOpen && pxSize >= 12
                    onVisibleChanged: requestPaint()
                    onPxSizeChanged: requestPaint()
                    onSrChanged: requestPaint()
                    onVrChanged: requestPaint()
                    onPaint: {
                        const g = getContext("2d")
                        g.reset()
                        if (!visible) return
                        const x0 = sr.x * src.width, y0 = sr.y * src.height     // 表示中の左上（元画像の画素）
                        const nx = Math.ceil(sr.width * src.width) + 1, ny = Math.ceil(sr.height * src.height) + 1
                        g.strokeStyle = Qt.rgba(0.5, 0.5, 0.5, 0.35)
                        g.lineWidth = 1
                        g.beginPath()
                        for (let i = 0; i <= nx; i++) {
                            const x = Math.round(vr.x + (Math.floor(x0) + i - x0) * pxSize) + 0.5
                            if (x < vr.x || x > vr.x + vr.width) continue
                            g.moveTo(x, vr.y); g.lineTo(x, vr.y + vr.height)
                        }
                        for (let j = 0; j <= ny; j++) {
                            const y = Math.round(vr.y + (Math.floor(y0) + j - y0) * pxSize) + 0.5
                            if (y < vr.y || y > vr.y + vr.height) continue
                            g.moveTo(vr.x, y); g.lineTo(vr.x + vr.width, y)
                        }
                        g.stroke()
                    }
                }

                // ---- スポイト：マウス位置の画素の値をカーソルの横に出す ----
                //   マウスボタンは受けない（パン・ズームはそのまま使える）。
                //   値は C++ が描画フレームごとに CUDA で読み、pickUpdated で届く
                Item {
                    id: eye
                    z: 4
                    anchors.fill: video
                    readonly property bool active: player.eyedropper && player.fileOpen && video.hasVideo && !player.encoding
                    readonly property rect vr: video.videoRect
                    readonly property rect sr: video.viewRect
                    readonly property int  mode: player.filterParams.eyedropperMode ?? 2
                    // 最新の値（C++ から描画フレームごとに更新）
                    readonly property var px: active ? player.pick : ({})
                    property point at: Qt.point(0, 0)

                    HoverHandler {
                        id: eyeHover
                        enabled: eye.active
                        cursorShape: Qt.CrossCursor
                        // 位置を渡すだけ。読み取りは C++ 側が描画フレームごとに行う
                        onPointChanged: { eye.at = point.position; eye.pick() }
                        onHoveredChanged: if (!hovered) player.pickPixel(-1, -1)   // 映像から外れたら読み取りを止める
                    }
                    onActiveChanged: if (!active) player.pickPixel(-1, -1)
                    // 一時停止中など描画が止まっている間も値を受け取るためのポーリング
                    Timer {
                        interval: 100; repeat: true
                        running: eye.active && eyeHover.hovered
                        onTriggered: eye.pick()
                    }
                    function pick() {
                        // 画面座標 → 映像上の位置（0..1）。映像外なら読み取りを止める
                        const fx = sr.x + (at.x - vr.x) / Math.max(1, vr.width)  * sr.width
                        const fy = sr.y + (at.y - vr.y) / Math.max(1, vr.height) * sr.height
                        const inside = at.x >= vr.x && at.x < vr.x + vr.width && at.y >= vr.y && at.y < vr.y + vr.height
                        if (inside) player.pickPixel(fx, fy); else player.pickPixel(-1, -1)
                    }
                    function luma(r, g, b) { return Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b) }

                    // 拡大時（1 画素が 6px 以上）は読み取り中の画素を枠で囲む
                    Rectangle {
                        readonly property real pxSize: video.sourceSize.width > 0 ? eye.vr.width / (eye.sr.width * video.sourceSize.width) : 0
                        visible: eye.active && eyeHover.hovered && eye.px.x !== undefined && pxSize >= 6
                        x: eye.vr.x + ((eye.px.x ?? 0) - eye.sr.x * video.sourceSize.width)  * pxSize
                        y: eye.vr.y + ((eye.px.y ?? 0) - eye.sr.y * video.sourceSize.height) * pxSize
                        width: pxSize; height: pxSize
                        color: "transparent"
                        border.width: 2
                        border.color: Theme.accent
                    }
                    Rectangle {
                        id: eyeCard
                        visible: eye.active && eyeHover.hovered && eye.px.x !== undefined
                        // カーソルの右下。はみ出すときは反対側へ
                        x: (eye.at.x + 18 + width  > eye.width)  ? eye.at.x - 18 - width  : eye.at.x + 18
                        y: (eye.at.y + 18 + height > eye.height) ? eye.at.y - 18 - height : eye.at.y + 18
                        width: eyeCol.implicitWidth + 20
                        height: eyeCol.implicitHeight + 16
                        radius: 8
                        color: Theme.overlay
                        border.color: Theme.border

                        Column {
                            id: eyeCol
                            x: 10; y: 8
                            spacing: 4
                            Text {
                                text: "X " + (eye.px.x ?? 0) + "   Y " + (eye.px.y ?? 0) + "   Frame " + (eye.px.frame ?? 0)
                                color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                            }
                            EyeRow { visible: eye.mode !== 1; label: "補正後"; r: eye.px.r ?? 0; g: eye.px.g ?? 0; b: eye.px.b ?? 0 }
                            EyeRow { visible: eye.mode !== 0; label: "補正前"; r: eye.px.r0 ?? 0; g: eye.px.g0 ?? 0; b: eye.px.b0 ?? 0 }
                        }
                    }
                }

                // ---- 比較表示の境目とラベル（「補正前 / 補正後」）----
                //   境目は映像上の位置なので viewRect から画面座標に変換する
                Item {
                    id: cmpLabels
                    z: 3          // 境目はドラッグ操作するのでパネルより上
                    anchors.fill: video
                    visible: player.compare && player.fileOpen && video.hasVideo && !player.encoding
                    readonly property bool horiz: (player.filterParams.compareDir ?? 0) === 0
                    readonly property real pos: (player.filterParams.comparePos ?? 500) / 1000
                    readonly property rect vr: video.videoRect
                    readonly property rect sr: video.viewRect
                    // 境目の画面上の位置（左右なら x、上下なら y）
                    readonly property real cut: horiz ? vr.x + (pos - sr.x) / Math.max(1e-6, sr.width)  * vr.width
                                                      : vr.y + (pos - sr.y) / Math.max(1e-6, sr.height) * vr.height
                    // 表示中の映像の枠（はみ出し防止）
                    readonly property real lo: horiz ? vr.x : vr.y
                    readonly property real hi: horiz ? vr.x + vr.width : vr.y + vr.height

                    // 境目をドラッグで動かす（前後 8px が掴める範囲）。
                    //   VideoItem のパン操作より手前にあるので、ここで横取りする
                    MouseArea {
                        id: cmpDrag
                        readonly property bool inView: cmpLabels.cut >= cmpLabels.lo && cmpLabels.cut <= cmpLabels.hi
                        visible: inView && player.controlsEnabled
                        x: cmpLabels.horiz ? cmpLabels.cut - 8 : cmpLabels.vr.x
                        y: cmpLabels.horiz ? cmpLabels.vr.y : cmpLabels.cut - 8
                        width:  cmpLabels.horiz ? 16 : cmpLabels.vr.width
                        height: cmpLabels.horiz ? cmpLabels.vr.height : 16
                        hoverEnabled: true
                        preventStealing: true
                        cursorShape: cmpLabels.horiz ? Qt.SplitHCursor : Qt.SplitVCursor
                        function setFrom(mx, my) {
                            const p = mapToItem(cmpLabels, mx, my)
                            const f = cmpLabels.horiz
                                ? cmpLabels.sr.x + (p.x - cmpLabels.vr.x) / Math.max(1, cmpLabels.vr.width)  * cmpLabels.sr.width
                                : cmpLabels.sr.y + (p.y - cmpLabels.vr.y) / Math.max(1, cmpLabels.vr.height) * cmpLabels.sr.height
                            player.setFilterParam("comparePos", Math.round(Math.max(0, Math.min(1, f)) * 1000))
                        }
                        onPressed: (m) => setFrom(m.x, m.y)
                        onPositionChanged: (m) => { if (pressed) setFrom(m.x, m.y) }
                        onDoubleClicked: player.setFilterParam("comparePos", 500)
                    }
                    // 掴み（境目の中央の丸）
                    Rectangle {
                        visible: cmpDrag.visible
                        width: 22; height: 22; radius: 11
                        x: cmpLabels.horiz ? cmpLabels.cut - 11 : cmpLabels.vr.x + cmpLabels.vr.width / 2 - 11
                        y: cmpLabels.horiz ? cmpLabels.vr.y + cmpLabels.vr.height / 2 - 11 : cmpLabels.cut - 11
                        color: cmpDrag.pressed ? Theme.accent : "#FFFFFF"
                        border.color: Qt.rgba(0, 0, 0, 0.35)
                        opacity: cmpDrag.containsMouse || cmpDrag.pressed ? 1.0 : 0.85
                        Icon {
                            anchors.centerIn: parent
                            size: 14
                            color: "#111316"
                            rotation: cmpLabels.horiz ? 0 : 90
                            paths: ["M9 7l-4 5 4 5", "M15 7l4 5-4 5"]
                        }
                    }

                    // 0 = 補正前（境目の手前）、1 = 補正後（境目の先）
                    Repeater {
                        model: 2
                        delegate: Rectangle {
                            required property int index
                            readonly property bool after: index === 1
                            width: chipText.implicitWidth + 16
                            height: 22
                            radius: 11
                            color: Theme.overlay
                            border.color: Theme.border
                            // 動画情報（左上）・ズーム / ヒストグラム（右側）と重ならない位置:
                            //   左右分割は下端、上下分割は横方向の中央
                            x: cmpLabels.horiz ? (after ? cmpLabels.cut + 8 : cmpLabels.cut - width - 8)
                                               : cmpLabels.vr.x + Math.min(cmpLabels.vr.width, cmpLabels.width) / 2 - width / 2
                            y: cmpLabels.horiz ? Math.min(cmpLabels.vr.y + cmpLabels.vr.height, cmpLabels.height) - height - 12
                                               : (after ? cmpLabels.cut + 8 : cmpLabels.cut - height - 8)
                            // 境目が端に寄ってラベルがはみ出すときは出さない
                            visible: after ? cmpLabels.cut + 8 + (cmpLabels.horiz ? width : height) <= cmpLabels.hi
                                           : cmpLabels.cut - 8 - (cmpLabels.horiz ? width : height) >= cmpLabels.lo
                            Text { id: chipText; anchors.centerIn: parent; text: parent.after ? "補正後" : "補正前"; color: Theme.text2; font.family: Theme.font; font.pixelSize: 11 }
                        }
                    }
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
                        text: "エンコード中"
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

            // 開くたびに作り直し、その時点の画面を 1 回だけ撮って固定する
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

    ScreenshotPanel {
        anchors.fill: parent
        anchors.topMargin: titleBar.visible ? titleBar.height : 0
        visible: win.shotOpen
        focus: visible
    }

    PropertiesPanel {
        anchors.fill: parent
        anchors.topMargin: titleBar.visible ? titleBar.height : 0
        visible: win.propsOpen
        focus: visible
        onCloseRequested: win.propsOpen = false
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
