import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// スポイト（映像の上にマウスを置くと、その画素の座標と値を表示）
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
            options: [{ text: "補正後", value: 0 }, { text: "補正前", value: 1 }, { text: "両方", value: 2 }]
            value: root.p.eyedropperMode
            onPicked: (v) => player.setFilterParam("eyedropperMode", v)
        }
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: "映像の上にマウスを置くと、その位置の座標・フレーム番号と、RGB 値（8bit）、輝度（Rec.709）を表示します。値は表示中のフレームから取得します。補正前はデコード直後の値、補正後は画像処理を適用した値です。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
