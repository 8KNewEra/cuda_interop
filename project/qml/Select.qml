import QtQuick
import QtQuick.Controls

// 選択肢 [{text, value}] と現在値 value で動くコンボボックス。
//   選べない候補は C++ 側でリストから外してあるので、ここは表示だけ。
ComboBox {
    id: root
    property var options: []
    property var value
    property string tip: ""
    signal picked(var v)

    function findIndex() {
        for (var i = 0; i < options.length; i++)
            if (options[i].value == value) return i   // 数値と文字列を同一視したいので ==
        return -1
    }

    model: options
    textRole: "text"
    currentIndex: findIndex()
    onActivated: (i) => { if (i >= 0 && i < options.length) root.picked(options[i].value) }

    implicitHeight: 36
    // ★ 幅は選択中の文字で変えない（レイアウト側で決める）。長い文字は省略表示。
    implicitWidth: 140
    hoverEnabled: true
    font.family: Theme.font
    font.pixelSize: 13
    opacity: enabled ? 1.0 : 0.45

    ToolTip.visible: hovered && tip.length > 0
    ToolTip.delay: 600
    ToolTip.text: tip

    background: Rectangle {
        radius: 8
        color: root.hovered && root.enabled ? Theme.hover : Theme.panel2
        border.color: root.activeFocus || root.popup.visible ? Theme.accent : Theme.border
    }

    contentItem: Text {
        leftPadding: 12
        rightPadding: 30
        text: root.currentIndex >= 0 ? root.displayText : "—"
        color: Theme.text
        font: root.font
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Icon {
        x: root.width - width - 10
        y: (root.height - height) / 2
        size: 14
        color: Theme.muted
        paths: ["M7 10l5 5 5-5"]
    }

    delegate: ItemDelegate {
        id: d
        required property var modelData
        required property int index
        width: ListView.view ? ListView.view.width : root.width
        height: 32
        highlighted: root.highlightedIndex === index
        contentItem: Text {
            text: d.modelData.text
            color: index === root.currentIndex ? Theme.accent : Theme.text
            font.family: Theme.font
            font.pixelSize: 13
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: 6
            color: d.highlighted ? Theme.hover : "transparent"
        }
    }

    popup: Popup {
        y: root.height + 4
        width: root.width
        implicitHeight: Math.min(contentItem.implicitHeight + 12, 320)
        padding: 6
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: root.popup.visible ? root.delegateModel : null
            currentIndex: root.highlightedIndex
            boundsBehavior: Flickable.StopAtBounds
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle {
            radius: 10
            color: Theme.panel2
            border.color: Theme.border
        }
    }
}
