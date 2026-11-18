import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// 画像処理パネル（右側のドロワー）
//   タブで「画像処理」と「表示ツール」を切り替える
//     画像処理  … 映像そのものを変える（ヒストグラム・画像保存・エンコードにも入る）
//     表示ツール … 画面表示だけ（ヒストグラム・画像保存・エンコードには入らない）
//   filters 配列 1 本で管理（group で振り分け）。key は PlayerController の同名 bool プロパティ
//   画像処理の並びは実際の適用順に合わせる:
//     画像補正・色補正・トーンカーブ（1 パス）→ ガウシアン → ソーベル → 平均化
Rectangle {
    id: root
    signal closeRequested()

    readonly property var filters: [
        { group: "proc", key: "imageAdjust",  name: "画像補正",           kind: "明るさ・コントラスト・ガンマ" },
        { group: "proc", key: "colorCorrect", name: "色補正",             kind: "彩度・色温度・RGB バランス" },
        { group: "proc", key: "toneCurve",    name: "トーンカーブ",       kind: "RGB・チャンネル別の階調補正" },
        { group: "proc", key: "gaussian",     name: "ガウシアンフィルタ", kind: "ぼかし・シャープ" },
        { group: "proc", key: "sobel",        name: "ソーベルフィルタ",   kind: "エッジ検出・ピーキング" },
        { group: "proc", key: "averaging",    name: "平均化フィルタ",     kind: "平滑化" },
        { group: "view", key: "compare",      name: "比較表示",           kind: "補正前後を並べて表示" },
        { group: "view", key: "channelView",  name: "チャンネル表示",     kind: "R・G・B・輝度を単色で表示" },
        { group: "view", key: "tileSeam",     name: "タイル境界チェック", kind: "タイルのつなぎ目を確認" },
        { group: "view", key: "eyedropper",   name: "スポイト",           kind: "画素の座標と値を表示" },
        { group: "view", key: "zebra",        name: "ゼブラ表示",         kind: "白飛び・黒つぶれを警告表示" }
    ]
    property string tab: "proc"                  // "proc" = 画像処理、"view" = 表示ツール
    property string expandedKey: ""

    function inGroup(g) { return filters.filter(f => f.group === g) }
    readonly property var shown: inGroup(tab)
    function countOn(g) {
        let n = 0
        for (const f of inGroup(g)) if (player[f.key]) n++
        return n
    }
    readonly property int activeCount: countOn(tab)
    readonly property int procCount: countOn("proc")
    readonly property int viewCount: countOn("view")

    // 同じグループで有効なものの中での順番（1 始まり）。無効なら 0
    function orderOf(key) {
        let n = 0
        for (const f of shown) {
            if (player[f.key]) n++
            if (f.key === key) return player[key] ? n : 0
        }
        return 0
    }
    // 今のタブのものだけ OFF
    function tabOff() { for (const f of shown) player[f.key] = false }

    width: 300
    radius: Theme.radius
    color: Theme.panel
    border.color: Theme.borderSoft
    clip: true

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ヘッダー
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 14
            Layout.leftMargin: 16
            Layout.bottomMargin: 10
            // タブ：画像処理 / 表示ツール（有効数をバッジで）
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 34
                radius: 9
                color: Theme.panel2
                border.color: Theme.border
                Row {
                    anchors.fill: parent
                    anchors.margins: 3
                    Repeater {
                        model: [{ key: "proc", text: "画像処理" }, { key: "view", text: "表示ツール" }]
                        delegate: AbstractButton {
                            id: tb
                            required property var modelData
                            readonly property bool on: root.tab === modelData.key
                            readonly property int  n: modelData.key === "proc" ? root.procCount : root.viewCount
                            width: parent.width / 2
                            height: parent.height
                            hoverEnabled: true
                            focusPolicy: Qt.NoFocus
                            onClicked: { root.tab = modelData.key; root.expandedKey = "" }
                            background: Rectangle { radius: 6; color: tb.on ? Theme.selected : (tb.hovered ? Theme.hover : "transparent") }
                            contentItem: Item {
                                Row {
                                    anchors.centerIn: parent
                                    spacing: 6
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: tb.modelData.text
                                        color: tb.on ? "#FFFFFF" : Theme.muted2
                                        font.family: Theme.font; font.pixelSize: 13; font.weight: tb.on ? Font.Bold : Font.Medium
                                    }
                                    Rectangle {
                                        visible: tb.n > 0
                                        anchors.verticalCenter: parent.verticalCenter
                                        width: 18; height: 18; radius: 9
                                        color: Theme.accent
                                        Text { anchors.centerIn: parent; text: tb.n; color: Theme.accentInk; font.family: Theme.mono; font.pixelSize: 11; font.bold: true }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            IconButton {
                implicitWidth: 36; implicitHeight: 36
                iconSize: 16
                paths: ["M6 6l12 12M18 6L6 18"]
                tip: "閉じる"
                onClicked: root.closeRequested()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            Layout.rightMargin: 12
            // タブごとの説明（エンコード・画像保存に入るかどうか）
            Text {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: root.tab === "proc"
                      ? "映像に適用され、画像保存・エンコードにも反映されます。上から順に処理します。"
                      : "表示のみに適用されます。画像保存・エンコードには反映されません。"
                color: root.tab === "proc" ? Theme.muted2 : Theme.accent
                font.family: Theme.font; font.pixelSize: 11
                lineHeight: 1.25
            }
            AbstractButton {
                id: offBtn
                text: "すべてオフ"
                enabled: player.controlsEnabled && root.activeCount > 0
                implicitHeight: 28
                implicitWidth: offLbl.implicitWidth + 16
                hoverEnabled: true
                focusPolicy: Qt.NoFocus
                onClicked: root.tabOff()
                background: Rectangle { radius: 6; color: offBtn.hovered && offBtn.enabled ? Theme.hover : "transparent" }
                contentItem: Text {
                    id: offLbl
                    text: offBtn.text
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    color: offBtn.enabled ? Theme.muted : Theme.faint
                    font.family: Theme.font; font.pixelSize: 12
                }
            }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: list.implicitHeight + 24
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            ColumnLayout {
                id: list
                x: 12; y: 8
                width: parent.width - 24
                spacing: 8

                Repeater {
                    model: root.shown
                    delegate: Rectangle {
                        id: card
                        required property var modelData
                        readonly property bool on: player[modelData.key]
                        readonly property bool expanded: root.expandedKey === modelData.key
                        readonly property int  order: root.orderOf(modelData.key)
                        // 係数を持つフィルタだけ（それ以外は null）
                        readonly property var  kernel: modelData.key === "gaussian"  ? player.gaussianKernel
                                                     : modelData.key === "averaging" ? player.averagingKernel : null

                        Layout.fillWidth: true
                        implicitHeight: cardCol.implicitHeight
                        radius: 10
                        color: Theme.panel2
                        border.color: on ? Qt.rgba(79/255, 209/255, 197/255, 0.45) : Theme.border

                        ColumnLayout {
                            id: cardCol
                            width: parent.width
                            spacing: 0

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.margins: 8
                                Layout.leftMargin: 12
                                spacing: 10

                                Rectangle {
                                    width: 22; height: 22; radius: 6
                                    color: card.on ? Theme.accent : Theme.border
                                    Text {
                                        anchors.centerIn: parent
                                        text: card.order > 0 ? card.order : "–"
                                        color: card.on ? Theme.accentInk : Theme.muted2
                                        font.family: Theme.mono; font.pixelSize: 12
                                    }
                                }

                                // 名前（クリックでパラメータ欄を開閉）
                                MouseArea {
                                    Layout.fillWidth: true
                                    implicitHeight: 36
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: root.expandedKey = card.expanded ? "" : card.modelData.key
                                    RowLayout {
                                        anchors.fill: parent
                                        spacing: 6
                                        ColumnLayout {
                                            spacing: 0
                                            Layout.fillWidth: true
                                            Text { Layout.fillWidth: true; elide: Text.ElideRight; text: card.modelData.name; color: card.on ? Theme.text : "#AEB4BD"; font.family: Theme.font; font.pixelSize: 13; font.weight: Font.Medium }
                                            Text { Layout.fillWidth: true; elide: Text.ElideRight; text: card.modelData.kind; color: Theme.muted2; font.family: Theme.font; font.pixelSize: 11 }
                                        }
                                        Icon {
                                            size: 14
                                            color: Theme.muted2
                                            paths: ["M9 6l6 6-6 6"]
                                            rotation: card.expanded ? 90 : 0
                                            Behavior on rotation { NumberAnimation { duration: 120 } }
                                        }
                                    }
                                }

                                ToggleSwitch {
                                    checked: card.on
                                    enabled: player.controlsEnabled
                                    onToggled: player[card.modelData.key] = checked
                                }
                            }

                            Rectangle {
                                visible: card.expanded
                                Layout.fillWidth: true
                                Layout.leftMargin: 12
                                Layout.rightMargin: 12
                                Layout.bottomMargin: 12
                                implicitHeight: (card.kernel ? kp.implicitHeight
                                                 : card.modelData.key === "sobel" ? pk.implicitHeight
                                                 : card.modelData.key === "zebra" ? zb.implicitHeight
                                                 : card.modelData.key === "imageAdjust" ? ia.implicitHeight
                                                 : card.modelData.key === "colorCorrect" ? cc.implicitHeight
                                                 : card.modelData.key === "channelView" ? ch.implicitHeight
                                                 : card.modelData.key === "compare" ? cmp.implicitHeight
                                                 : card.modelData.key === "toneCurve" ? tc.implicitHeight
                                                 : card.modelData.key === "tileSeam" ? ts.implicitHeight
                                                 : card.modelData.key === "eyedropper" ? ed.implicitHeight
                                                 : paramText.implicitHeight) + 20
                                radius: 8
                                color: "#0F1113"
                                border.color: Theme.border
                                // 係数を持つフィルタ（ガウシアン / 平均化）
                                KernelParams {
                                    id: kp
                                    visible: card.kernel !== null
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    kernel: card.kernel
                                    title: card.modelData.name
                                    enabled: player.controlsEnabled && card.on   // フィルタ OFF の間は設定も触れない
                                }
                                PeakingParams {
                                    id: pk
                                    visible: card.modelData.key === "sobel"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on   // 同上
                                }
                                ZebraParams {
                                    id: zb
                                    visible: card.modelData.key === "zebra"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on   // 同上
                                }
                                ImageAdjustParams {
                                    id: ia
                                    visible: card.modelData.key === "imageAdjust"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                ColorParams {
                                    id: cc
                                    visible: card.modelData.key === "colorCorrect"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                EyedropperParams {
                                    id: ed
                                    visible: card.modelData.key === "eyedropper"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                TileSeamParams {
                                    id: ts
                                    visible: card.modelData.key === "tileSeam"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                ToneCurveParams {
                                    id: tc
                                    visible: card.modelData.key === "toneCurve"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                CompareParams {
                                    id: cmp
                                    visible: card.modelData.key === "compare"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                ChannelParams {
                                    id: ch
                                    visible: card.modelData.key === "channelView"
                                    x: 10; y: 10
                                    width: parent.width - 20
                                    enabled: player.controlsEnabled && card.on
                                }
                                Text {
                                    id: paramText
                                    visible: card.kernel === null && !pk.visible && !zb.visible && !cc.visible && !ch.visible && !ia.visible && !cmp.visible && !tc.visible && !ts.visible && !ed.visible
                                    anchors.fill: parent
                                    anchors.margins: 10
                                    wrapMode: Text.WordWrap
                                    text: "設定項目はありません"
                                    color: Theme.muted2; font.family: Theme.font; font.pixelSize: 12
                                    lineHeight: 1.3
                                }
                            }
                        }
                    }
                }

            }
        }
    }
}
