import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ゼブラ表示の設定（白飛び / 黒つぶれ）
//   表示専用：ヒストグラム・画像保存・エンコードには入らない
ColumnLayout {
    id: root
    readonly property var p: player.filterParams
    spacing: 10

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Rectangle { width: 10; height: 10; radius: 2; color: "#000000"; border.color: Theme.muted; opacity: enabled ? 1 : 0.4 }
        Text { Layout.fillWidth: true; text: "白飛び警告"; color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ToggleSwitch { checked: root.p.zebraHighOn; onToggled: player.setFilterParam("zebraHighOn", checked) }
    }
    MiniSlider {
        Layout.fillWidth: true
        enabled: root.p.zebraHighOn
        label: "レベル"
        fieldWidth: 58
        from: 50; to: 100; unit: "%"
        value: root.p.zebraHighLevel
        onMoved: (v) => player.setFilterParam("zebraHighLevel", v)
    }

    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSoft }

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Rectangle { width: 10; height: 10; radius: 2; color: "#2673FF"; opacity: enabled ? 1 : 0.4 }
        Text { Layout.fillWidth: true; text: "黒つぶれ警告"; color: enabled ? Theme.text2 : Theme.faint; font.family: Theme.font; font.pixelSize: 12 }
        ToggleSwitch { checked: root.p.zebraLowOn; onToggled: player.setFilterParam("zebraLowOn", checked) }
    }
    MiniSlider {
        Layout.fillWidth: true
        enabled: root.p.zebraLowOn
        label: "レベル"
        fieldWidth: 58
        from: 0; to: 50; unit: "%"
        value: root.p.zebraLowLevel
        onMoved: (v) => player.setFilterParam("zebraLowLevel", v)
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: "白飛びはレベル以上、黒つぶれはレベル以下の領域に縞模様を表示します。輝度は Rec.709 の係数で算出します。"
        color: enabled ? Theme.muted2 : Theme.faint; font.family: Theme.font; font.pixelSize: 11
        lineHeight: 1.3
    }
}
