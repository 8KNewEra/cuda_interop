import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 画像処理パネルの「係数」欄（ガウシアン / 平均化）
//   kernel = FilterKernel（player.gaussianKernel / player.averagingKernel）
//   サイズ・σ から自動生成。係数の直接編集は KernelEditor（ポップアップ）
ColumnLayout {
    id: root
    property var kernel: null
    property string title: ""
    readonly property bool gaussian: kernel && kernel.kind === "gaussian"
    readonly property int n: kernel ? kernel.size : 0

    spacing: 10

    component SmallButton: AbstractButton {
        id: b
        implicitHeight: 30
        Layout.fillWidth: true
        hoverEnabled: true
        focusPolicy: Qt.NoFocus
        opacity: enabled ? 1.0 : 0.4
        background: Rectangle { radius: 7; color: b.hovered && b.enabled ? Theme.hover : Theme.panel2; border.color: Theme.border }
        contentItem: Text {
            text: b.text
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12
        }
    }


    function fmt(v) {
        // 0.000123 のような小さい値も読めるように有効数字 3 桁
        if (v === 0) return "0"
        const a = Math.abs(v)
        return a >= 0.01 ? v.toFixed(3) : v.toPrecision(2)
    }
    readonly property real maxAbs: {
        let m = 0
        if (kernel) for (const v of kernel.coeffs) m = Math.max(m, Math.abs(v))
        return m > 0 ? m : 1
    }

    // ---- 処理（ガウシアンのみ）：ぼかし / シャープ化 ----
    //   シャープ化 = アンシャープマスク（元 ＋ 強さ ×（元 − ぼかし））。ぼかし量はサイズと σ
    readonly property var fp: player.filterParams
    RowLayout {
        visible: root.gaussian
        Layout.fillWidth: true
        spacing: 8
        Text { text: "処理"; Layout.preferredWidth: 44; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Segmented {
            Layout.fillWidth: true
            options: [{ text: "ぼかし", value: 0 }, { text: "シャープ", value: 1 }]
            value: root.fp.gaussianMode
            onPicked: (v) => player.setFilterParam("gaussianMode", v)
        }
    }
    MiniSlider {
        visible: root.gaussian && root.fp.gaussianMode === 1
        Layout.fillWidth: true
        labelWidth: 44
        fieldWidth: 58
        label: "強さ"
        from: 0; to: 500; unit: "%"; defaultValue: 100
        value: root.fp.sharpenAmount
        onMoved: (v) => player.setFilterParam("sharpenAmount", v)
    }

    // ---- サイズ（セグメント）----
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text { text: "サイズ"; Layout.preferredWidth: 44; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 30
            radius: 7
            color: Theme.panel2
            border.color: Theme.border
            opacity: enabled ? 1.0 : 0.4
            Row {
                anchors.fill: parent
                anchors.margins: 2
                Repeater {
                    model: root.kernel ? root.kernel.sizeOptions : []
                    delegate: AbstractButton {
                        id: seg
                        required property var modelData
                        readonly property bool on: root.n === modelData.value
                        width: (parent.width) / 4
                        height: parent.height
                        hoverEnabled: true
                        focusPolicy: Qt.NoFocus
                        onClicked: root.kernel.setSize(modelData.value)
                        background: Rectangle { radius: 5; color: seg.on ? Theme.selected : (seg.hovered ? Theme.hover : "transparent") }
                        contentItem: Text {
                            text: seg.modelData.text
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            color: seg.on ? "#FFFFFF" : Theme.muted2
                            font.family: Theme.mono; font.pixelSize: 11
                        }
                    }
                }
            }
        }
    }

    // ---- σ（ガウシアンのみ）----
    RowLayout {
        visible: root.gaussian
        Layout.fillWidth: true
        spacing: 8
        Text { text: "σ"; Layout.preferredWidth: 44; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        NumberSpin {
            Layout.fillWidth: true
            implicitHeight: 30
            decimals: 2
            from: 10; to: 1000; stepSize: 5
            value: root.kernel ? root.kernel.sigma100 : 0
            onMoved: (v) => root.kernel.setSigma100(v)
        }
    }

    // ---- 係数のプレビュー（濃さ = 重み）----
    Rectangle {
        Layout.fillWidth: true
        implicitHeight: width
        radius: 6
        opacity: enabled ? 1.0 : 0.4
        color: "#0B0C0E"
        border.color: Theme.border

        Grid {
            id: grid
            anchors.fill: parent
            anchors.margins: 4
            columns: root.n
            spacing: 2
            readonly property real cell: root.n > 0 ? (width - spacing * (root.n - 1)) / root.n : 0
            Repeater {
                model: root.n * root.n
                delegate: Rectangle {
                    required property int index
                    readonly property real modelData: root.kernel ? (root.kernel.coeffs[index] ?? 0) : 0
                    width: grid.cell; height: grid.cell
                    radius: 3
                    // 正は accent、負は marker（エッジ系の係数を入れたとき用）
                    color: modelData >= 0 ? Qt.rgba(79/255, 209/255, 197/255, 0.08 + 0.72 * Math.abs(modelData) / root.maxAbs)
                                          : Qt.rgba(242/255, 165/255, 65/255, 0.08 + 0.72 * Math.abs(modelData) / root.maxAbs)
                    Text {
                        anchors.centerIn: parent
                        visible: root.n <= 5
                        text: root.fmt(modelData)
                        color: Theme.text
                        font.family: Theme.mono; font.pixelSize: root.n <= 3 ? 12 : 10
                    }
                }
            }
        }
    }

    // ---- 合計・正規化 ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            Layout.fillWidth: true
            text: "合計 " + (root.kernel ? root.kernel.sum.toFixed(4) : "-")
                  + (root.kernel && root.kernel.custom ? "  · 編集済み" : "")
            color: root.kernel && root.kernel.custom ? Theme.marker : Theme.muted2
            font.family: Theme.mono; font.pixelSize: 11
        }
        Text { text: "正規化"; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ToggleSwitch {
            checked: root.kernel ? root.kernel.normalize : false
            onToggled: root.kernel.setNormalize(checked)
            ToolTip.visible: hovered
            ToolTip.delay: 500
            ToolTip.text: "係数の合計が 1 になるよう補正し、明るさを維持します。"
        }
    }

    // ---- 操作 ----
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        SmallButton { text: "係数を編集…"; onClicked: editor.open() }
        SmallButton {
            text: "リセット"
            enabled: root.kernel && root.kernel.custom
            onClicked: root.kernel.reset()
        }
    }

    KernelEditor {
        id: editor
        kernel: root.kernel
        title: root.title
    }
}
