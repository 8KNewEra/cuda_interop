import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 性能 HUD（映像の左下。Ctrl+Shift+P / 表示トグルの「性能」で ON/OFF）
//   1 フレームの処理を段に分けた時間（直近 0.25 秒の平均と最大）と、フレーム間隔の推移を出す
//   値は PlayerController::perf（0.25 秒ごと。中身は PerfMeter::take、perfmeter.h）
//     段: decode / convert / filter / display / hist / cpu（各 {avg, max, n}。n = 0 は「—」）
//     マルチ GPU デコード: compress / transfer / expand と内訳 decodeParts / expandParts、nvdecs、frozen
//     （意味は docs/DECODE_CORE.md の 8.9）
//     interval（フレーム間隔の {avg, max}）、intervals（フレーム間隔の列）、budget（1 フレームの持ち時間 ms）
//   バーの長さは「持ち時間」に対する割合。持ち時間を超えると赤。使っていない段（表示ツール・ヒストグラムが OFF）は
//   「—」（行は消さない。消すと高さが変わってちらつく）。再生中以外は直前の値で止める（frozen、「停止中 ·」）
//   画面に収まらない高さなら縦スクロール
Rectangle {
    id: root
    property var perf: ({})
    property bool fileOpen: false
    property string fileKey: ""

    readonly property real budget: perf.budget || 0
    readonly property var  iv: perf.interval || ({})

    // 幅は文字の大きさ（環境のフォント・拡大率）に合わせて広げる。最小 320
    implicitWidth: Math.max(320, labelW + 2 * valW + 3 * 8 + 90 + 28)
    // 段の列幅: ラベル列と数値列（一番長い文字が収まる幅。Stage に渡す）
    readonly property real labelW: Math.max(96, Math.ceil(Math.max(tmLabel.width, tmSub.width)) + 4)
    readonly property real valW:   Math.max(44, Math.ceil(tmVal.width) + 4)
    TextMetrics { id: tmLabel; font.family: Theme.font; font.pixelSize: 12; text: "▾ デコード（待ち）" }
    TextMetrics { id: tmSub;   font.family: Theme.font; font.pixelSize: 12; text: "  └ 差分戻し・書き込み" }
    TextMetrics { id: tmVal;   font.family: Theme.mono; font.pixelSize: 12; text: "000.00" }

    // デコード（待ち）の内訳を開いているか（普段は畳む。行をクリックで開閉）
    property bool showDecodeParts: false
    readonly property bool hasDecodeParts: (perf.decodeParts || []).length > 0
    // 展開の内訳を開いているか（同じく普段は畳む）
    property bool showExpandParts: false
    readonly property bool hasExpandParts: (perf.expandParts || []).length > 0
    implicitHeight: col.implicitHeight + 24
    radius: Theme.radius
    color: Theme.overlay
    border.color: Theme.border

    // ---- フレーム間隔の推移（直近 histLen フレーム）----
    readonly property int histLen: 300
    property var hist: []
    onPerfChanged: {
        if (perf.frozen) return   // 再生していない間は直前の値のまま（推移も足さない）
        const add = perf.intervals || []
        if (add.length === 0) return
        const b = hist.concat(add)
        hist = b.length > histLen ? b.slice(b.length - histLen) : b   // 作り直して代入（変更通知のため）
    }
    onFileKeyChanged: hist = []
    readonly property real histMax: {
        let m = 0
        for (const v of hist) if (v > m) m = v
        return m
    }

    // NVDEC 使用率（%。PlayerController::perf。player.gpus は動画情報が出ている間しか更新されないので使わない）
    //   nvdec: デコードに使っている GPU の最大（-1 = 取れない）、nvdecs: GPU ごと [{id, dec, tiles, engines}]（表示用 GPU が先頭）
    readonly property int nvdec: perf.nvdec !== undefined ? perf.nvdec : -1
    readonly property var nvdecs: perf.nvdecs || []

    function ms(v) { return (v === undefined || v === null || v < 0) ? "—" : Number(v).toFixed(v < 10 ? 2 : 1) }
    function over(v) { return root.budget > 0 && v > root.budget }
    // フレーム間隔は再生が fps に合わせて進むので、持ち時間とほぼ同じになるのが正常。
    //   わずかな揺れで色が付かないよう、5% を超えたときだけ「超えた」とみなす
    function overIv(v) { return root.budget > 0 && v > root.budget * 1.05 }

    // デコードの上限 fps の目安 = 今デコードしている fps ÷ 動いている NVDEC の使用率（GPU ごとに出して一番低いもの）
    //   NVDEC は先読みしながら並行して動くので、デコードの時間（待ち）では限界が分からない。
    //   使用率は GPU の NVDEC 全基の平均。1 本のストリーム（タイル）は 1 基しか使えないので、
    //   受け持つタイル数がエンジン数より少ない GPU は「動いている基」の使用率に直してから割る
    //   （例: 2 基の GPU で 1 タイル、使用率 50% → 動いている 1 基は 100%）。エンジン数が分からなければ 1 基とみなす
    //   使用率が低い・再生していない（間隔が取れない）ときは出さない（-1）
    readonly property real nvdecLimitFps: {
        if (!(iv.n > 0) || !(perf.fps > 0)) return -1
        let lim = -1
        for (const d of nvdecs) {
            if (!(d.dec >= 10)) continue
            const e = d.engines > 0 ? d.engines : 1
            const s = d.tiles > 0 ? d.tiles : e
            const busy = (d.dec / 100) * e / Math.min(s, e)
            const v = perf.fps / busy
            if (lim < 0 || v < lim) lim = v
        }
        if (lim < 0 && nvdec >= 10) lim = perf.fps / (nvdec / 100)
        return lim
    }
    // タイル数がエンジン数より少ない（NVDEC の一部が遊んでいる）GPU
    readonly property var idleNvdec: nvdecs.filter(d => d.engines > 1 && d.tiles > 0 && d.tiles < d.engines)
    // マルチ GPU デコードの転送の上限 = 1000 ÷ 1 フレームの転送時間（1 本の経路で順に送るので、それ以上は出ない）
    //   圧縮転送のときは、別 GPU で前のフレームを送りながら次を圧縮する（重なる）ので、
    //   圧縮・送る・展開（表示用 GPU）のうち一番遅いもので決まる
    readonly property real transferLimitFps: {
        const t = perf.transfer
        if (!(t && t.n > 0 && t.avg > 0)) return -1
        const c = perf.compress, e = perf.expand
        const slow = Math.max(t.avg, (c && c.n > 0 && c.avg > 0) ? c.avg : 0, (e && e.n > 0 && e.avg > 0) ? e.avg : 0)
        return 1000 / slow
    }
    // 上限の目安 = 小さい方。どちらで決まっているかも出す（limitBy: "NVDEC" / "転送"）
    readonly property real decodeLimitFps: {
        const a = nvdecLimitFps, b = transferLimitFps
        if (a > 0 && b > 0) return Math.min(a, b)
        return a > 0 ? a : b
    }
    readonly property string limitBy: transferLimitFps > 0 && (nvdecLimitFps <= 0 || transferLimitFps < nvdecLimitFps) ? "転送" : "NVDEC"

    // 段の 1 行: ラベル | バー（平均 / 持ち時間）| 平均 ms | 最大 ms
    //   inline component から外側の id（root）は参照しない（Qt 6.9 で解決できないことがある）ので、
    //   持ち時間は budget で受け取り、書式の関数も中に持つ
    component Stage: RowLayout {
        id: st
        property string label
        property var    v: ({})
        property real   budget: 0
        property real   labelWidth: 96   // 列幅（文字の大きさに合わせて HUD 側で決める）
        property real   valueWidth: 44
        readonly property real avg: (v && v.n > 0) ? v.avg : -1
        readonly property real mx:  (v && v.n > 0) ? v.max : -1
        function ms(x) { return x < 0 ? "—" : Number(x).toFixed(x < 10 ? 2 : 1) }
        function over(x) { return budget > 0 && x > budget }
        Layout.fillWidth: true
        spacing: 8
        Text { text: st.label; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.preferredWidth: st.labelWidth; elide: Text.ElideRight }
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 6
            radius: 3
            color: Theme.border
            Rectangle {
                height: parent.height
                radius: 3
                width: st.budget > 0 && st.avg > 0 ? Math.min(1, st.avg / st.budget) * parent.width : 0
                color: st.over(st.avg) ? "#FF7A7A" : (st.avg > st.budget * 0.75 ? Theme.marker : Theme.accent)
            }
        }
        Text {
            text: st.ms(st.avg)
            color: st.over(st.avg) ? "#FF7A7A" : Theme.text2
            font.family: Theme.mono; font.pixelSize: 12
            horizontalAlignment: Text.AlignRight
            Layout.preferredWidth: st.valueWidth
        }
        Text {
            text: st.ms(st.mx)
            color: st.over(st.mx) ? "#FF7A7A" : Theme.faint
            font.family: Theme.mono; font.pixelSize: 11
            horizontalAlignment: Text.AlignRight
            Layout.preferredWidth: st.valueWidth
        }
    }

    // 収まりきらないとき（内訳を開いた・GPU が多い・小さなウィンドウ）は中をスクロールする（動画情報と同じ）
    //   高さは呼び出し側で min(implicitHeight, 表示できる高さ) にする
    Flickable {
        id: flick
        anchors.fill: parent
        anchors.topMargin: 12
        anchors.bottomMargin: 12
        clip: true
        contentWidth: width
        contentHeight: col.implicitHeight
        // 収まっているときはホイール・ドラッグを受けない（映像のズーム・パンに渡す）
        interactive: contentHeight > height + 1
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { policy: flick.interactive ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff }

        ColumnLayout {
            id: col
            x: 14
            width: flick.width - 28 - (flick.interactive ? 8 : 0)   // スクロールバーと数字が重ならないように
            spacing: 6

            // 見出し: 性能 / 描画 fps（目標）
            RowLayout {
                Layout.fillWidth: true
                Text { text: "性能"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.bold: true }
                Item { Layout.fillWidth: true }
                Text {
                    text: (perf.frozen ? "停止中 · " : "") + (perf.fps !== undefined ? Number(perf.fps).toFixed(1) : "—") + " fps"
                          + (perf.videoFps > 0 ? "  / 目標 " + Number(perf.videoFps).toFixed(2) : "")
                    color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                }
            }

            // 列見出し
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Text { text: "ms / フレーム"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; Layout.preferredWidth: root.labelW }
                Text {
                    Layout.fillWidth: true
                    text: root.budget > 0 ? "持ち時間 " + root.ms(root.budget) + " ms" : ""
                    color: Theme.faint; font.family: Theme.font; font.pixelSize: 11
                }
                Text { text: "平均"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: root.valW }
                Text { text: "最大"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: root.valW }
            }

            // デコードはスレッドが待った時間（NVDEC の忙しさは下の使用率と上限 fps で見る）
            //   内訳があるときはクリックで開閉（▸ / ▾）
            Stage {
                label: (root.hasDecodeParts ? (root.showDecodeParts ? "▾ " : "▸ ") : "") + "デコード（待ち）"
                v: root.perf.decode; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW
                TapHandler { enabled: root.hasDecodeParts; onTapped: root.showDecodeParts = !root.showDecodeParts }
                HoverHandler { enabled: root.hasDecodeParts; cursorShape: Qt.PointingHandCursor }
            }
            // その内訳（マルチタイルの GPU デコードのとき。CPU の実時間）。行の数は GPU の数で決まる
            Repeater {
                model: root.showDecodeParts ? (root.perf.decodeParts || []).length : 0
                delegate: Stage {
                    required property int index
                    readonly property var p: (root.perf.decodeParts || [])[index] || ({})
                    label: "  └ " + (p.label || "")
                    v: p
                    budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW
                    opacity: 0.85
                }
            }
            Stage { label: "変換";         v: root.perf.convert; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW }
            Stage { label: "画像処理";     v: root.perf.filter;  budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW }
            Stage { label: "表示ツール";   v: root.perf.display; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW }
            Stage { label: "ヒストグラム"; v: root.perf.hist;    budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW }
            Stage { label: "描画スレッド"; v: root.perf.cpu;     budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW }
            // 圧縮転送（nvCOMP）のときだけ: 別 GPU での差分の前処理と圧縮
            Stage { label: "圧縮"; v: root.perf.compress; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW; visible: root.nvdecs.length > 1 && (root.perf.xferRatio || -1) > 0 }
            // マルチ GPU デコードのときだけ: 別 GPU → メインメモリの転送（その GPU の時間。圧縮転送なら圧縮後のデータを送る時間）
            Stage { label: "転送"; v: root.perf.transfer; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW; visible: root.nvdecs.length > 1 }
            // 圧縮転送（nvCOMP）のときだけ: 表示用 GPU での取り込み・展開
            //   内訳（取り込み / nvCOMP の展開 / 差分戻し）はクリックで開閉
            Stage {
                label: (root.hasExpandParts ? (root.showExpandParts ? "▾ " : "▸ ") : "") + "展開"
                v: root.perf.expand; budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW
                visible: root.nvdecs.length > 1 && (root.perf.xferRatio || -1) > 0
                TapHandler { enabled: root.hasExpandParts; onTapped: root.showExpandParts = !root.showExpandParts }
                HoverHandler { enabled: root.hasExpandParts; cursorShape: Qt.PointingHandCursor }
            }
            Repeater {
                model: root.showExpandParts && root.nvdecs.length > 1 ? (root.perf.expandParts || []).length : 0
                delegate: Stage {
                    required property int index
                    readonly property var p: (root.perf.expandParts || [])[index] || ({})
                    label: "  └ " + (p.label || "")
                    v: p
                    budget: root.budget; labelWidth: root.labelW; valueWidth: root.valW
                    opacity: 0.85
                }
            }
            Text {
                visible: root.nvdecs.length > 1 && (root.perf.transferGbps || -1) > 0
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                wrapMode: Text.Wrap
                // 圧縮転送のときの速度は「元の大きさ ÷ 圧縮と転送の遅い方の時間」（圧縮で見かけの帯域が増えた分を含む）
                text: ((root.perf.xferRatio || -1) > 0 ? "圧縮率 " + Number(root.perf.xferRatio).toFixed(2) + " · 実効 " : "転送速度 ")
                      + Number(root.perf.transferGbps).toFixed(2) + " GB/s"
                color: Theme.faint; font.family: Theme.mono; font.pixelSize: 11
            }

            Rectangle { Layout.fillWidth: true; Layout.topMargin: 2; height: 1; color: Theme.borderSoft }

            // フレーム間隔（= 実際のフレーム時間）と推移。破線 = 持ち時間
            RowLayout {
                Layout.fillWidth: true
                Text { text: "フレーム間隔"; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12 }
                Item { Layout.fillWidth: true }
                Text {
                    text: root.ms(iv.n > 0 ? iv.avg : -1) + " ms  最大 " + root.ms(iv.n > 0 ? iv.max : -1)
                    // 平均が持ち時間を超えている = 目標の fps が出ていない（赤）、最大だけ超えた = 時々カクつく（橙）
                    color: root.overIv(iv.n > 0 ? iv.avg : -1) ? "#FF7A7A"
                         : root.overIv(iv.n > 0 ? iv.max : -1) ? Theme.marker : Theme.text2
                    font.family: Theme.mono; font.pixelSize: 12
                }
            }
            Sparkline {
                Layout.fillWidth: true
                Layout.preferredHeight: 44
                capacity: root.histLen
                values: root.hist
                refValue: root.budget > 0 ? root.budget : -1
                maxValue: Math.max(root.budget * 2, root.histMax * 1.1, 1)
                lineColor: Theme.accent
            }

            // NVDEC の使用率と、そこから見たデコードの上限 fps（デコードが詰まっているかの目安）
            //   1 枚なら「NVDEC（表示 GPU）」、マルチ GPU デコードなら GPU ごとの値を並べる（例 GPU0 78% / GPU1 60%）
            //   上限は使用率がいちばん高い GPU で決まる。目標に近い（目標の 1.1 倍未満）ときは橙
            RowLayout {
                Layout.fillWidth: true
                visible: root.nvdec >= 0
                Text {
                    text: root.nvdecs.length > 1 ? "NVDEC" : "NVDEC（表示 GPU）"
                    color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12
                }
                Text {
                    Layout.fillWidth: true
                    horizontalAlignment: Text.AlignRight
                    wrapMode: Text.Wrap   // GPU が多いと 1 行に収まらない
                    text: root.nvdecs.length > 1
                          ? root.nvdecs.map(d => "GPU" + d.id + "\u00a0" + d.dec + "%").join(" / ")   // GPU と値の間では折り返さない
                          : root.nvdec + " %"
                    color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
                }
            }
            // NVDEC の一部が遊んでいる GPU（1 タイル = 1 基なので、エンジン数より少ないタイルでは全基を使えない）
            Text {
                visible: root.idleNvdec.length > 0
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                horizontalAlignment: Text.AlignRight
                // 例「NVDEC 1/2 基のみ使用: GPU3・0・1・2」（使っている基 / 全基 が同じ GPU をまとめる）
                text: {
                    const g = {}
                    for (const d of root.idleNvdec) {
                        const k = d.tiles + "/" + d.engines
                        ;(g[k] = g[k] || []).push(d.id)
                    }
                    return Object.keys(g).map(k => "NVDEC " + k + " 基のみ使用: GPU" + g[k].join("・")).join("  ")
                }
                color: Theme.marker; font.family: Theme.font; font.pixelSize: 11
            }
            RowLayout {
                Layout.fillWidth: true
                visible: root.decodeLimitFps > 0
                Text { text: "デコード上限（目安）"; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12 }
                Item { Layout.fillWidth: true }
                Text {
                    text: "約 " + Math.round(root.decodeLimitFps) + " fps" + (root.nvdecs.length > 1 ? "（" + root.limitBy + "）" : "")
                    color: root.perf.videoFps > 0 && root.decodeLimitFps < root.perf.videoFps * 1.1 ? Theme.marker : Theme.text2
                    font.family: Theme.mono; font.pixelSize: 12
                }
            }
        }
    }
}
