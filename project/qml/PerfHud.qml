import QtQuick
import QtQuick.Layouts

// 性能 HUD（映像の左下。Ctrl+Shift+P / 表示トグルの「性能」で ON/OFF）
//   1 フレームの処理を段に分けた時間（直近 0.25 秒の平均と最大）と、フレーム間隔の推移を出す
//   値は PlayerController::perf（0.25 秒ごと。中身は PerfMeter::take、perfmeter.h）
//     段: decode / convert / filter / display / hist / cpu（各 {avg, max, n}。n = 0 は「—」）
//     interval（フレーム間隔の {avg, max}）、intervals（フレーム間隔の列）、budget（1 フレームの持ち時間 ms）
//   バーの長さは「持ち時間」に対する割合。持ち時間を超えると赤。使っていない段（表示ツール・ヒストグラムが OFF、
//   一時停止中）は「—」（行は消さない。消すと 0.25 秒ごとに高さが変わってちらつく）
Rectangle {
    id: root
    property var perf: ({})
    property bool fileOpen: false
    property string fileKey: ""

    readonly property real budget: perf.budget || 0
    readonly property var  iv: perf.interval || ({})

    width: 320
    implicitHeight: col.implicitHeight + 24
    radius: Theme.radius
    color: Theme.overlay
    border.color: Theme.border

    // ---- フレーム間隔の推移（直近 histLen フレーム）----
    readonly property int histLen: 300
    property var hist: []
    onPerfChanged: {
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
    //   nvdec: デコードに使っている GPU の最大（-1 = 取れない）、nvdecs: GPU ごと [{id, dec}]（表示用 GPU が先頭）
    readonly property int nvdec: perf.nvdec !== undefined ? perf.nvdec : -1
    readonly property var nvdecs: perf.nvdecs || []

    function ms(v) { return (v === undefined || v === null || v < 0) ? "—" : Number(v).toFixed(v < 10 ? 2 : 1) }
    function over(v) { return root.budget > 0 && v > root.budget }
    // フレーム間隔は再生が fps に合わせて進むので、持ち時間とほぼ同じになるのが正常。
    //   わずかな揺れで色が付かないよう、5% を超えたときだけ「超えた」とみなす
    function overIv(v) { return root.budget > 0 && v > root.budget * 1.05 }

    // デコードの上限 fps の目安 = 今デコードしている fps ÷ NVDEC 使用率
    //   NVDEC は先読みしながら並行して動くので、デコードの時間（待ち）では限界が分からない。
    //   使用率が低い・再生していない（間隔が取れない）ときは出さない（-1）
    readonly property real nvdecLimitFps: {
        if (!(nvdec >= 10) || !(iv.n > 0) || !(perf.fps > 0)) return -1
        return perf.fps / (nvdec / 100)
    }
    // マルチ GPU デコードの転送の上限 = 1000 ÷ 1 フレームの転送時間（1 本の経路で順に送るので、それ以上は出ない）
    readonly property real transferLimitFps: {
        const t = perf.transfer
        return (t && t.n > 0 && t.avg > 0) ? 1000 / t.avg : -1
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
        readonly property real avg: (v && v.n > 0) ? v.avg : -1
        readonly property real mx:  (v && v.n > 0) ? v.max : -1
        function ms(x) { return x < 0 ? "—" : Number(x).toFixed(x < 10 ? 2 : 1) }
        function over(x) { return budget > 0 && x > budget }
        Layout.fillWidth: true
        spacing: 8
        Text { text: st.label; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12; Layout.preferredWidth: 96 }
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
            Layout.preferredWidth: 44
        }
        Text {
            text: st.ms(st.mx)
            color: st.over(st.mx) ? "#FF7A7A" : Theme.faint
            font.family: Theme.mono; font.pixelSize: 11
            horizontalAlignment: Text.AlignRight
            Layout.preferredWidth: 44
        }
    }

    ColumnLayout {
        id: col
        x: 14; y: 12
        width: root.width - 28
        spacing: 6

        // 見出し: 性能 / 描画 fps（目標）
        RowLayout {
            Layout.fillWidth: true
            Text { text: "性能"; color: Theme.text; font.family: Theme.font; font.pixelSize: 13; font.bold: true }
            Item { Layout.fillWidth: true }
            Text {
                text: (perf.fps !== undefined ? Number(perf.fps).toFixed(1) : "—") + " fps"
                      + (perf.videoFps > 0 ? "  / 目標 " + Number(perf.videoFps).toFixed(2) : "")
                color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
            }
        }

        // 列見出し
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Text { text: "ms / フレーム"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; Layout.preferredWidth: 96 }
            Text {
                Layout.fillWidth: true
                text: root.budget > 0 ? "持ち時間 " + root.ms(root.budget) + " ms" : ""
                color: Theme.faint; font.family: Theme.font; font.pixelSize: 11
            }
            Text { text: "平均"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: 44 }
            Text { text: "最大"; color: Theme.faint; font.family: Theme.font; font.pixelSize: 11; horizontalAlignment: Text.AlignRight; Layout.preferredWidth: 44 }
        }

        // デコードはスレッドが待った時間（NVDEC の忙しさは下の使用率と上限 fps で見る）
        Stage { label: "デコード（待ち）"; v: root.perf.decode;  budget: root.budget }
        Stage { label: "変換";         v: root.perf.convert; budget: root.budget }
        Stage { label: "画像処理";     v: root.perf.filter;  budget: root.budget }
        Stage { label: "表示ツール";   v: root.perf.display; budget: root.budget }
        Stage { label: "ヒストグラム"; v: root.perf.hist;    budget: root.budget }
        Stage { label: "描画スレッド"; v: root.perf.cpu;     budget: root.budget }
        // マルチ GPU デコードのときだけ: 別 GPU → メインメモリの転送（その GPU の時間。右に速度）
        Stage { label: "転送"; v: root.perf.transfer; budget: root.budget; visible: root.nvdecs.length > 1 }
        Text {
            visible: root.nvdecs.length > 1 && (root.perf.transferGbps || -1) > 0
            Layout.alignment: Qt.AlignRight
            text: "転送速度 " + Number(root.perf.transferGbps).toFixed(2) + " GB/s"
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
            Item { Layout.fillWidth: true }
            Text {
                text: root.nvdecs.length > 1
                      ? root.nvdecs.map(d => "GPU" + d.id + " " + d.dec + "%").join(" / ")
                      : root.nvdec + " %"
                color: Theme.text2; font.family: Theme.mono; font.pixelSize: 12
            }
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
