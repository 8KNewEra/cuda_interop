import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// エンコード設定（再生画面の上に重ねて出す）
//   値と制約はすべて EncodeController（context property "encoder"）が持つ。
//   ここは表示と入力だけ。
Item {
    id: root
    signal closeRequested()

    readonly property bool locked: encoder.encoding

    // ---- 部品 ----
    component KeyLabel: Text {
        property string tip: ""
        Layout.preferredWidth: 132
        color: Theme.muted
        font.family: Theme.font
        font.pixelSize: 13
        elide: Text.ElideRight
        ToolTip.visible: keyHover.containsMouse && tip.length > 0
        ToolTip.delay: 600
        ToolTip.text: tip
        MouseArea { id: keyHover; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
    }

    component Card: Rectangle {
        id: card
        property string title: ""
        property Item headerExtra: null
        default property alias body: inner.data
        Layout.fillWidth: true
        implicitHeight: cardCol.implicitHeight + 36
        radius: Theme.radius
        color: Theme.panel
        border.color: Theme.borderSoft
        ColumnLayout {
            id: cardCol
            anchors.fill: parent
            anchors.margins: 18
            anchors.leftMargin: 20
            anchors.rightMargin: 20
            spacing: 14
            RowLayout {
                Layout.fillWidth: true
                Text { text: card.title; color: Theme.text; font.family: Theme.font; font.pixelSize: 14; font.bold: true }
                Item { Layout.fillWidth: true }
                Item {
                    implicitWidth: card.headerExtra ? card.headerExtra.implicitWidth : 0
                    implicitHeight: card.headerExtra ? card.headerExtra.implicitHeight : 0
                    data: card.headerExtra ? [card.headerExtra] : []
                }
            }
            ColumnLayout {
                id: inner
                Layout.fillWidth: true
                spacing: 8
            }
        }
    }

    component PlainButton: AbstractButton {
        id: pb
        property bool primary: false
        implicitHeight: 40
        implicitWidth: pbText.implicitWidth + 40
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

    component Labeled: ColumnLayout {
        property string label
        property string tip
        property alias options: sel.options
        property alias value: sel.value
        property alias selectEnabled: sel.enabled
        signal picked(var v)
        // ★ 2 列を常に同じ幅にする（選んだ文字の長さで列幅が変わらないように）
        Layout.fillWidth: true
        Layout.preferredWidth: 100
        Layout.minimumWidth: 100
        spacing: 6
        KeyLabel { text: parent.label; tip: parent.tip; Layout.preferredWidth: -1 }
        Select { id: sel; Layout.fillWidth: true; Layout.preferredWidth: 100; tip: parent.tip; onPicked: (v) => parent.picked(v) }
    }

    component InfoLine: RowLayout {
        property string text
        property color dot: Theme.accent
        spacing: 8
        Rectangle { width: 6; height: 6; radius: 3; color: parent.dot }
        Text { Layout.fillWidth: true; text: parent.text; wrapMode: Text.Wrap; color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12 }
    }

    component Th: Text { color: Theme.muted; font.family: Theme.font; font.pixelSize: 12 }

    // ---- 暗幕（後ろの再生画面は触れない）----
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.6)
        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons; onWheel: (w) => w.accepted = true }
    }

    Keys.onEscapePressed: if (!root.locked) root.closeRequested()

    // 開くたびに高さの記録をやり直す（素材が変わると GPU 表やタイルの項目が変わる）
    //   ★ 列数（幅で 1 列 / 2 列が変わる）が変わったときも測り直す。
    //     開いた直後はレイアウトが何度か組み直され、途中で実際より高い値が
    //     出る。少し待って落ち着いてから測り直す（それまでは今の高さに追従）。
    function resetPeak() { sheet.peakBody = body.implicitHeight }
    Timer { id: settleTimer; interval: 150; onTriggered: root.resetPeak() }
    onVisibleChanged: if (visible) { resetPeak(); settleTimer.restart() }
    Connections {
        target: body
        function onImplicitHeightChanged() {
            if (settleTimer.running) sheet.peakBody = body.implicitHeight
            else sheet.peakBody = Math.max(sheet.peakBody, body.implicitHeight)
        }
        function onColumnsChanged() { settleTimer.restart() }
    }

    // ---- 本体 ----
    Rectangle {
        id: sheet
        // ★ 縦は「中身が全部入る高さ」で止める（大きな画面で下に空白を作らない）。
        //   入りきらないときだけウィンドウいっぱいまで使い、中をスクロールさせる。
        //   中身の高さはレート制御などで行が出入りすると変わるので、開いている間の
        //   最大値（peakBody）で決める。設定を切り替えるたびに枠が伸び縮みしない。
        //   位置は縦方向も中央。高さは最大値でしか変わらないので、ほぼ動かない。
        property real peakBody: 0
        readonly property real chromeH: 60 + 72 + 40      // ヘッダー + フッター + 余白
        anchors.horizontalCenter: parent.horizontalCenter
        y: Math.max(24, Math.round((parent.height - height) / 2))
        width: Math.min(parent.width - 48, 1240)
        height: Math.min(parent.height - 48, chromeH + peakBody)
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
                        paths: ["M4 17V7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10", "M2 19h20", "M10 9l4 3-4 3z"]
                    }
                    Text { text: "エンコード設定"; color: Theme.text; font.family: Theme.font; font.pixelSize: 18; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: encoder.summary
                        color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                        elide: Text.ElideLeft
                        Layout.maximumWidth: sheet.width * 0.5
                    }
                    IconButton {
                        implicitWidth: 40; implicitHeight: 40
                        iconSize: 18
                        enabled: !root.locked
                        paths: ["M6 6l12 12M18 6L6 18"]
                        tip: root.locked ? "エンコード中は閉じられません" : "閉じる（Esc）"
                        onClicked: root.closeRequested()
                    }
                }
            }

            // 設定本体（スクロール）
            Flickable {
                id: flick
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: body.implicitHeight + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: flick.contentHeight > flick.height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff }

                GridLayout {
                    id: body
                    x: 24; y: 20
                    width: flick.width - 48
                    columns: width >= 900 ? 2 : 1
                    columnSpacing: 16
                    rowSpacing: 16
                    enabled: !root.locked

                    // ===== 左カラム =====
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: 1
                        spacing: 16

                        Card {
                            title: "ファイル名"
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12
                                KeyLabel { text: "保存パス" }
                                Rectangle {
                                    Layout.fillWidth: true
                                    implicitHeight: 36
                                    radius: 8
                                    color: "#0F1113"
                                    border.color: Theme.border
                                    Text {
                                        anchors.fill: parent
                                        anchors.leftMargin: 10
                                        anchors.rightMargin: 10
                                        verticalAlignment: Text.AlignVCenter
                                        text: encoder.path
                                        elide: Text.ElideMiddle
                                        color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                                    }
                                    ToolTip.visible: pathHover.containsMouse
                                    ToolTip.delay: 600
                                    ToolTip.text: encoder.path
                                    MouseArea { id: pathHover; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
                                }
                                PlainButton { text: "保存先…"; implicitHeight: 36; onClicked: encoder.browsePath() }
                            }
                            CheckBox {
                                id: sepBox
                                text: "タイルごとに別ファイルへ出力"
                                checked: encoder.separateFiles
                                onToggled: encoder.separateFiles = checked
                                font.family: Theme.font
                                font.pixelSize: 13
                                ToolTip.visible: hovered
                                ToolTip.delay: 600
                                ToolTip.text: "out.mp4 → out.tile0.mp4 / out.tile1.mp4 …  音声はタイル 0 のファイルに入ります"
                                indicator: Rectangle {
                                    x: sepBox.leftPadding
                                    y: (sepBox.height - height) / 2
                                    width: 18; height: 18; radius: 4
                                    color: sepBox.checked ? Theme.accent : Theme.panel2
                                    border.color: sepBox.checked ? Theme.accent : Theme.dashed
                                    Icon { anchors.centerIn: parent; visible: sepBox.checked; size: 14; strokeWidth: 2.4; color: Theme.accentInk; paths: ["M5 12l5 5 9-10"] }
                                }
                                contentItem: Text {
                                    leftPadding: sepBox.indicator.width + 10
                                    text: sepBox.text
                                    color: Theme.text2
                                    font: sepBox.font
                                    verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }

                        Card {
                            title: "一般設定"
                            // ★ 録画フォーマット → コーデックの順（コンテナで選べるコーデックが変わる）
                            RowLayout { Layout.fillWidth: true; spacing: 12
                                KeyLabel { text: "録画フォーマット"; tip: "出力ファイルの拡張子で muxer が決まります" }
                                Select { Layout.fillWidth: true; options: encoder.containerOptions; value: encoder.container; onPicked: (v) => encoder.setValue("container", v) } }
                            RowLayout { Layout.fillWidth: true; spacing: 12
                                KeyLabel { text: "コーデック"; tip: "-c:v" }
                                Select { Layout.fillWidth: true; options: encoder.codecOptions; value: encoder.codec; tip: "-c:v"; onPicked: (v) => encoder.setValue("codec", v) } }
                            RowLayout { Layout.fillWidth: true; spacing: 12
                                KeyLabel { text: "フレームレート"; tip: "-r" }
                                Select { Layout.fillWidth: true; options: encoder.framerateOptions; value: encoder.framerate; tip: "-r"; onPicked: (v) => encoder.setValue("framerate", v) } }
                            RowLayout { Layout.fillWidth: true; spacing: 12
                                KeyLabel { text: "分割エンコード"; tip: "-split_encode_mode" }
                                Select { Layout.fillWidth: true; options: encoder.splitOptions; value: encoder.split; tip: "-split_encode_mode"; onPicked: (v) => encoder.setValue("split", v) } }
                            RowLayout { Layout.fillWidth: true; spacing: 12
                                KeyLabel { text: "タイル分割"; tip: "キャンバスを分割して複数の NVENC で並列にエンコードします" }
                                Select { Layout.fillWidth: true; options: encoder.tileOptions; value: encoder.tile; onPicked: (v) => encoder.setValue("tile", v) } }
                        }

                        Card {
                            title: "音声を含める"
                            enabled: encoder.audioAvailable
                            opacity: enabled ? 1.0 : 0.5
                            headerExtra: Row {
                                spacing: 8
                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: encoder.audioAvailable ? "" : "この動画には音声がありません"
                                    color: Theme.muted; font.family: Theme.font; font.pixelSize: 12
                                }
                                ToggleSwitch {
                                    anchors.verticalCenter: parent.verticalCenter
                                    checked: encoder.audioEnabled && encoder.audioAvailable
                                    onToggled: encoder.audioEnabled = checked
                                }
                            }
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                enabled: encoder.audioEnabled
                                opacity: enabled ? 1.0 : 0.45
                                RowLayout { Layout.fillWidth: true; spacing: 12
                                    KeyLabel { text: "音声コーデック"; tip: "-c:a" }
                                    Select { Layout.fillWidth: true; options: encoder.audioCodecOptions; value: encoder.audioCodec; onPicked: (v) => encoder.setValue("audioCodec", v) } }
                                ValueRow {
                                    Layout.fillWidth: true
                                    visible: !encoder.audioLossless
                                    label: "音声ビットレート"; tip: "-b:a"
                                    from: 64; to: 512; stepSize: 16
                                    value: encoder.audioBitrate
                                    unit: "kbps"
                                    onMoved: (v) => encoder.setAudioBitrate(v)
                                }
                                ValueRow {
                                    Layout.fillWidth: true
                                    label: "ゲイン"; tip: "0 で素通し。float のうちに掛けます"
                                    from: -30; to: 30; stepSize: 1
                                    value: encoder.audioGain
                                    unit: "dB"
                                    onMoved: (v) => encoder.setAudioGain(v)
                                }
                            }
                        }
                    }

                    // ===== 右カラム：詳細設定 =====
                    Card {
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: 1.3
                        title: "詳細設定"

                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2
                            columnSpacing: 20
                            rowSpacing: 10
                            Labeled { label: "プリセット"; tip: "-preset"; options: encoder.presetOptions; value: encoder.preset; onPicked: (v) => encoder.setValue("preset", v) }
                            Labeled { label: "チューニング"; tip: "-tune"; options: encoder.tuneOptions; value: encoder.tune; onPicked: (v) => encoder.setValue("tune", v) }
                            Labeled { label: "キーフレーム間隔"; tip: "-g"; options: encoder.gopOptions; value: encoder.gop; onPicked: (v) => encoder.setValue("gop", v) }
                            Labeled { label: "B フレーム"; tip: encoder.bframeEditable ? "-bf" : "-bf（この GPU は B フレーム非対応）"; options: encoder.bframeOptions; value: encoder.bframes; selectEnabled: encoder.bframeEditable; onPicked: (v) => encoder.setValue("bframes", v) }
                            Labeled { label: "レート制御"; tip: "-rc"; options: encoder.rcOptions; value: encoder.rc; onPicked: (v) => encoder.setValue("rc", v) }
                            Labeled { label: "マルチパス"; tip: "-multipass"; options: encoder.passOptions; value: encoder.pass; onPicked: (v) => encoder.setValue("pass", v) }
                        }

                        Rectangle { Layout.fillWidth: true; Layout.topMargin: 6; Layout.bottomMargin: 2; height: 1; color: Theme.borderSoft }

                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: "ビットレート / 品質"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.bold: true }
                            Item { Layout.fillWidth: true }
                            KeyLabel { visible: encoder.showBasis; text: "ビットレート基準"; Layout.preferredWidth: -1
                                       tip: "ビットレートを 1 ストリームあたりで指定するか、ファイル全体の合計で指定するか" }
                            Select { visible: encoder.showBasis; Layout.preferredWidth: 200; options: encoder.basisOptions; value: encoder.basis; onPicked: (v) => encoder.setValue("basis", v) }
                        }

                        ValueRow {
                            Layout.fillWidth: true
                            visible: encoder.showTarget
                            label: "目標ビットレート"; tip: "-b:v"
                            from: encoder.targetMin; to: encoder.targetMax
                            value: encoder.targetRate
                            unit: encoder.rateUnit
                            onMoved: (v) => encoder.setTargetRate(v)
                        }
                        ValueRow {
                            Layout.fillWidth: true
                            visible: encoder.showMax
                            label: "最大ビットレート"; tip: "-maxrate"
                            from: encoder.maxMin; to: encoder.maxMax
                            value: encoder.maxRate
                            unit: encoder.rateUnit
                            onMoved: (v) => encoder.setMaxRate(v)
                        }
                        ValueRow {
                            Layout.fillWidth: true
                            visible: encoder.showQuality
                            label: encoder.qualityLabel; tip: encoder.qualityFlag
                            from: 1; to: 51
                            value: encoder.quality
                            onMoved: (v) => encoder.setQuality(v)
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.topMargin: 4
                            implicitHeight: infoCol.implicitHeight + 24
                            radius: 10
                            color: "#0F1113"
                            border.color: Theme.borderSoft
                            ColumnLayout {
                                id: infoCol
                                anchors.fill: parent
                                anchors.margins: 12
                                anchors.leftMargin: 14
                                spacing: 6
                                InfoLine { visible: encoder.rateInfoTarget.length > 0; text: encoder.rateInfoTarget }
                                InfoLine { visible: encoder.rateInfoMax.length > 0; text: encoder.rateInfoMax; dot: Theme.histB }
                                InfoLine { text: encoder.vramInfo; dot: Theme.marker }
                            }
                        }
                    }

                    // ===== マルチ GPU（全幅）=====
                    Card {
                        Layout.columnSpan: body.columns
                        title: "マルチGPU 重み付け設定"
                        enabled: encoder.multiGpu
                        opacity: enabled || root.locked ? 1.0 : 0.6
                        headerExtra: Text {
                            text: encoder.multiGpu || root.locked ? "重みに応じてタイルを各 GPU の NVENC に割り当てます" : "GPU が 1 枚のため使用しません"
                            color: Theme.muted; font.family: Theme.font; font.pixelSize: 12
                        }

                        // 見出し
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12
                            Th { text: "GPU Name"; Layout.fillWidth: true }
                            Th { text: "Encoder Usage"; Layout.preferredWidth: 200 }
                            Th { text: "Memory(MB)"; Layout.preferredWidth: 140; horizontalAlignment: Text.AlignRight }
                            Th { text: "Detail"; Layout.preferredWidth: 150; Layout.leftMargin: 24 }   // Memory の数字と離す
                            Th { text: "Weight"; Layout.preferredWidth: 130 }
                            Th { text: "Tiles"; Layout.preferredWidth: 170; horizontalAlignment: Text.AlignRight }
                        }
                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

                        Repeater {
                            model: encoder.gpus
                            delegate: ColumnLayout {
                                id: gpuRow
                                required property var modelData
                                required property int index
                                // 1 秒ごとに更新される実測値（行そのものは作り直さない）
                                readonly property var stats: (encoder.gpuStats && encoder.gpuStats[index]) || ({})
                                Layout.fillWidth: true
                                spacing: 0
                                RowLayout {
                                    Layout.fillWidth: true
                                    Layout.topMargin: 6
                                    Layout.bottomMargin: 6
                                    spacing: 12
                                    opacity: modelData.used ? 1.0 : 0.5
                                    Text { Layout.fillWidth: true; text: modelData.name + " (" + modelData.id + ")"; elide: Text.ElideRight; color: Theme.text; font.family: Theme.font; font.pixelSize: 13 }
                                    // エンコーダ（NVENC）使用率：動画情報の GPU バーと同じ見た目
                                    RowLayout {
                                        // 中のバーが fillWidth なので、列幅を明示的に固定する
                                        Layout.fillWidth: false
                                        Layout.preferredWidth: 200
                                        Layout.maximumWidth: 200
                                        spacing: 10
                                        ToolTip.visible: usageHover.hovered
                                        ToolTip.delay: 400
                                        ToolTip.text: "NVENC " + (gpuRow.stats.encUsage ?? 0) + "% / GPU 全体 " + (gpuRow.stats.gpuUsage ?? 0) + "%"
                                        HoverHandler { id: usageHover }
                                        Rectangle {
                                            Layout.fillWidth: true
                                            height: 4; radius: 2
                                            color: Theme.hover
                                            Rectangle {
                                                width: parent.width * Math.min(1, (gpuRow.stats.encUsage ?? 0) / 100)
                                                height: parent.height; radius: 2
                                                color: Theme.accent
                                                Behavior on width { NumberAnimation { duration: 300; easing.type: Easing.OutCubic } }
                                            }
                                        }
                                        Text {
                                            Layout.preferredWidth: 40
                                            horizontalAlignment: Text.AlignRight
                                            text: (gpuRow.stats.encUsage ?? 0) + "%"
                                            color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                                        }
                                    }
                                    Text {
                                        Layout.preferredWidth: 140; horizontalAlignment: Text.AlignRight
                                        text: gpuRow.stats.memUsed !== undefined ? gpuRow.stats.memUsed + " / " + gpuRow.stats.memMax : modelData.memory
                                        color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                                    }
                                    Item {
                                        Layout.preferredWidth: 150
                                        Layout.leftMargin: 24   // 見出しと揃える
                                        implicitHeight: 22
                                        Rectangle {
                                            visible: modelData.primary
                                            anchors.verticalCenter: parent.verticalCenter
                                            width: primLbl.implicitWidth + 16; height: 22; radius: 11
                                            color: Qt.rgba(79/255, 209/255, 197/255, 0.14)
                                            Text { id: primLbl; anchors.centerIn: parent; text: "Primary (OpenGL)"; color: Theme.accent; font.family: Theme.font; font.pixelSize: 11 }
                                        }
                                    }
                                    // 重み（− 値 ＋）
                                    Rectangle {
                                        Layout.preferredWidth: 130
                                        implicitHeight: 32
                                        radius: 8
                                        color: Theme.panel2
                                        border.color: Theme.border
                                        opacity: modelData.locked ? 0.4 : 1.0
                                        Row {
                                            anchors.fill: parent
                                            anchors.margins: 1
                                            IconButton {
                                                width: 36; height: parent.height
                                                implicitWidth: 36; implicitHeight: 30
                                                iconSize: 14
                                                paths: ["M6 12h12"]
                                                enabled: !modelData.locked && modelData.weight > modelData.minWeight
                                                autoRepeat: true
                                                onClicked: encoder.setGpuWeight(modelData.id, modelData.weight - 1)
                                            }
                                            Text {
                                                width: parent.width - 72; height: parent.height
                                                horizontalAlignment: Text.AlignHCenter
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.weight
                                                color: Theme.text; font.family: Theme.mono; font.pixelSize: 13
                                            }
                                            IconButton {
                                                width: 36; height: parent.height
                                                implicitWidth: 36; implicitHeight: 30
                                                iconSize: 14
                                                paths: ["M6 12h12", "M12 6v12"]
                                                enabled: !modelData.locked && modelData.weight < 10
                                                autoRepeat: true
                                                onClicked: encoder.setGpuWeight(modelData.id, modelData.weight + 1)
                                            }
                                        }
                                    }
                                    Text { Layout.preferredWidth: 170; horizontalAlignment: Text.AlignRight; text: modelData.status; color: modelData.used ? Theme.text : Theme.muted; font.family: Theme.mono; font.pixelSize: 12 }
                                }
                                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSoft }
                            }
                        }
                    }
                }
            }

            // フッター：進捗と実行
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 72
                color: Theme.bar
                Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Theme.borderSoft }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 16

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                                // エンコード中は速度も出す（例：エンコード中 (62.01 fps)）
                                text: encoder.encoding && encoder.encodeFps > 0
                                      ? encoder.statusText + " (" + encoder.encodeFps.toFixed(2) + " fps)"
                                      : (encoder.statusText.length > 0 ? encoder.statusText : "待機中")
                                color: encoder.finishedOk ? Theme.accent
                                     : (encoder.encoding || encoder.statusText.length === 0 ? Theme.muted : Theme.marker)
                                font.family: Theme.font; font.pixelSize: 12
                                font.weight: encoder.finishedOk ? Font.Bold : Font.Normal
                            }
                            Text {
                                text: encoder.progressValue + " / " + encoder.progressMax + " フレーム  "
                                      + Math.floor(encoder.progressValue * 100 / Math.max(1, encoder.progressMax)) + "%"
                                color: Theme.muted; font.family: Theme.mono; font.pixelSize: 12
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 6; radius: 3
                            color: Theme.hover
                            Rectangle {
                                width: parent.width * Math.min(1, encoder.progressValue / Math.max(1, encoder.progressMax))
                                height: 6; radius: 3
                                color: encoder.finishedOk || encoder.encoding ? Theme.accent : Theme.muted2
                            }
                        }
                    }

                    PlainButton { text: "キャンセル"; enabled: encoder.encoding; onClicked: encoder.cancel() }
                    PlainButton { primary: true; text: "エンコード開始"; enabled: !encoder.encoding; onClicked: encoder.start() }
                }
            }
        }
    }
}
