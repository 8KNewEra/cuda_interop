import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 動画情報オーバーレイ（映像の左上。Main.qml から player の値を渡す）
//   並び: 基本情報（ファイル・コーデック・解像度・ビット深度・音声）→ 描画フレームレート
//         → GPU ごとの使用率（プライマリが先頭。GPU / 3D / CUDA / NVDEC / MEM）
//   info / gpus は player が表示中 500ms ごとに更新（PlayerController::videoInfo / gpus）。
//   GPU の値は info_thread が 1 秒ごとに取得（NVML ＋ Windows の PDH）。
//   推移グラフ（直近 60 秒）の記録はこのファイルの中だけで持つ（C++ 側は今の値だけ）。
Rectangle {
    id: root
    property var info: ({})
    property var gpus: []
    // フレーム番号は毎フレーム変わるので info（500ms 更新）とは別に受け取る
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

    // ラベル ＋ 推移グラフ ＋ 今の値 の 1 行（GPU 欄）
    component Spark: RowLayout {
        id: sp
        property string label
        property var    values: []
        property string valueText
        property color  lineColor: Theme.accent
        Layout.fillWidth: true
        spacing: 8
        Text { text: sp.label; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.preferredWidth: 56 }
        Sparkline { Layout.fillWidth: true; Layout.preferredHeight: 18; values: sp.values; lineColor: sp.lineColor }
        Text { text: sp.valueText; color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: 96 }
    }

    // プライマリ（OpenGL を描いている GPU）を先頭に、残りは CUDA の番号順
    readonly property var gpuList: gpus.slice().sort((a, b) => (b.primary - a.primary) || (a.id - b.id))

    // ---- 推移の記録（1 秒ごと、直近 histLen 秒）----
    //   GPU は id ごとに {usage, g3d, cuda, dec, mem}（%）、描画 fps は再生中だけ（停止中は欠け）
    readonly property int histLen: 60
    property var hist: ({})
    property var fpsHist: []
    function pushed(a, v) { const b = a ? a.slice(-(histLen - 1)) : []; b.push(v); return b }
    function sample() {
        const h = {}
        for (const g of root.gpus) {
            const o = root.hist[g.id] || {}
            h[g.id] = {
                usage: pushed(o.usage, g.usage),
                g3d:   pushed(o.g3d,  g.g3d  >= 0 ? g.g3d  : null),
                cuda:  pushed(o.cuda, g.cuda >= 0 ? g.cuda : null),
                dec:   pushed(o.dec,  g.dec),
                mem:   pushed(o.mem,  g.memMax > 0 ? 100 * g.memUsed / g.memMax : 0)
            }
        }
        root.hist = h
        root.fpsHist = pushed(root.fpsHist, root.playing ? root.renderFps : null)
    }
    Timer { interval: 1000; running: true; repeat: true; onTriggered: root.sample() }
    // ファイルを閉じた・別のファイルを開いたら記録をやり直す
    property bool fileOpen: false
    readonly property string fileKey: fileOpen ? (info.fileName || "") : ""
    onFileKeyChanged: { hist = ({}); fpsHist = [] }

    function fmtFps(v) { return (v === undefined || v === null) ? "-" : Number(v).toFixed(2) }

    // 収まりきらないとき（小さなウィンドウ・GPU が多いとき）は中をスクロールする
    //   高さは呼び出し側で min(implicitHeight, 表示できる高さ) にする
    Flickable {
        id: flick
        anchors.fill: parent
        anchors.topMargin: 14
        anchors.bottomMargin: 14
        clip: true
        contentWidth: width
        contentHeight: col.implicitHeight
        // 収まっているときはホイール・ドラッグを受けない（映像のズーム・パンに渡す）
        interactive: contentHeight > height + 1
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { policy: flick.interactive ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff }   // 続きがあると分かるよう常に出す

        ColumnLayout {
            id: col
            x: 16
            width: flick.width - 32
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
                Key { text: "Bit Depth" }        Val { text: root.info.pixel || "-" }
                Key { text: "Video Framerate" }  Num { text: root.fmtFps(root.info.fps) }
                Key { text: "Max Frame" }        Num { text: root.info.maxFrame !== undefined ? root.info.maxFrame : "-" }
                Key { text: "FrameNo" }          Num { text: root.frameNo }
                // 音声: コーデック · サンプリングレート · チャンネル数（ダウンミックス時は「元 → 再生」）
                Key { text: "Audio"; visible: !!root.info.hasAudio }
                Val { text: root.info.audio || ((root.info.audioChannels || 0) + "ch"); visible: !!root.info.hasAudio }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

            // 描画フレームレート: OpenGL で実際に表示できた枚数を目標と並べて出す
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
                              ? '<font color="' + root.renderColor + '">' + Number(root.renderFps).toFixed(2) + '</font>'
                                + ' / ' + Number(root.targetFps).toFixed(2) + ' fps'
                              : '一時停止中'
                        color: Theme.muted2; font.family: Theme.mono; font.pixelSize: 12
                    }
                }
                // 直近 60 秒の推移（破線 = 目標）。縦軸は目標の 1.25 倍か実測の最大の大きい方
                Sparkline {
                    capacity: root.histLen
                    Layout.fillWidth: true
                    Layout.preferredHeight: 36
                    values: root.fpsHist
                    refValue: root.targetFps
                    lineColor: root.playing ? root.renderColor : Theme.muted2
                    maxValue: Math.max(root.targetFps * 1.25,
                                       ...root.fpsHist.map(v => v === null ? 0 : v), 1)
                }
                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    text: "実測 / 目標（動画 " + root.fmtFps(root.info.fps) + " fps × 再生速度 " + root.speed + "x）"
                    color: Theme.faint; font.family: Theme.font; font.pixelSize: 11
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

            Repeater {
                // モデルは台数だけにする（gpus は更新のたびに新しい配列になるので、そのまま渡すと
                //   毎回すべての行が作り直されてグラフがちらつく）。中身は gpuList[index] から読む
                model: root.gpuList.length
                delegate: ColumnLayout {
                    id: gpuCol
                    required property int index
                    readonly property var modelData: root.gpuList[index] || ({})
                    readonly property var hs: root.hist[modelData.id] || ({})
                    Layout.fillWidth: true
                    spacing: 4

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

                    // 全体 / 3D（OpenGL 描画）/ CUDA / NVDEC（デコード）/ メモリ。直近 60 秒の推移
                    //   3D・CUDA は取れない環境では隠す。CUDA のエンジンが無い環境（HAGS 有効時など）は
                    //   CUDA の負荷が 3D に入るので「3D+CUDA」と表示する
                    Spark { label: "GPU"; values: gpuCol.hs.usage || []; valueText: modelData.usage + "%" }
                    Spark {
                        visible: modelData.g3d >= 0
                        label: modelData.cuda >= 0 ? "3D" : "3D+CUDA"
                        values: gpuCol.hs.g3d || []; valueText: modelData.g3d + "%"
                    }
                    Spark {
                        visible: modelData.cuda !== undefined && modelData.cuda >= 0
                        label: "CUDA"; values: gpuCol.hs.cuda || []; valueText: modelData.cuda + "%"
                    }
                    Spark { label: "NVDEC"; values: gpuCol.hs.dec || []; valueText: modelData.dec + "%" }
                    Spark {
                        label: "MEM"; lineColor: Theme.muted2
                        values: gpuCol.hs.mem || []
                        valueText: modelData.memUsed + "/" + modelData.memMax + " MB"
                    }
                }
            }
        }
    }
}
