import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// チャンネル表示（R / G / B / 輝度 のどれか 1 つを白黒で表示）
//   表示専用：ヒストグラム・画像保存・エンコードには入らない
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    spacing: 10

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text { text: "表示"; Layout.preferredWidth: 44; color: enabled ? Theme.muted : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        Segmented {
            Layout.fillWidth: true
            options: [{ text: "R", value: 0 }, { text: "G", value: 1 }, { text: "B", value: 2 }, { text: "輝度", value: 3 }]
            value: root.p.channelMode
            onPicked: (v) => player.setFilterParam("channelMode", v)
        }
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: "選択したチャンネルをモノクロで表示します。輝度は Rec.709 の係数で算出します。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
