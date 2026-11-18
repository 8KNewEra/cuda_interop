import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 比較表示（補正前 / 補正後を 1 画面で並べる）
//   表示専用：ヒストグラム・画像保存・エンコードには入らない
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    spacing: 10

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text { text: "分割"; Layout.preferredWidth: 44; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Segmented {
            Layout.fillWidth: true
            options: [{ text: "左右", value: 0 }, { text: "上下", value: 1 }]
            value: root.p.compareDir
            onPicked: (v) => player.setFilterParam("compareDir", v)
        }
    }

    MiniSlider {
        Layout.fillWidth: true
        labelWidth: 44
        fieldWidth: 58
        label: "位置"
        from: 0; to: 1000; decimals: 1; unit: "%"; defaultValue: 500
        value: root.p.comparePos
        onMoved: (v) => player.setFilterParam("comparePos", v)
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: (root.p.compareDir === 0 ? "左" : "上") + "に補正前、"
              + (root.p.compareDir === 0 ? "右" : "下") + "に補正後を表示します。境界線は映像上でドラッグして移動でき、ダブルクリックで中央に戻ります。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
