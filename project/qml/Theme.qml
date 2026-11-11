pragma Singleton
import QtQuick

// モック（Design キャンバス）の色・文字をそのまま定数にしたもの
QtObject {
    readonly property color bg:        "#0B0C0E"
    readonly property color bar:       "#111316"
    readonly property color panel:     "#141619"
    readonly property color panel2:    "#1B1E22"
    readonly property color hover:     "#22262C"
    readonly property color selected:  "#2E333A"
    readonly property color border:    "#2A2E35"
    readonly property color borderSoft:"#22262C"
    readonly property color dashed:    "#3A3F47"
    readonly property color overlay:   Qt.rgba(17/255, 19/255, 22/255, 0.82)

    readonly property color text:      "#E8EAED"
    readonly property color text2:     "#C9CED6"
    readonly property color muted:     "#9BA3AE"
    readonly property color muted2:    "#8F97A2"
    readonly property color faint:     "#6B727C"

    readonly property color accent:    "#4FD1C5"
    readonly property color accentInk:  "#06201D"
    readonly property color marker:    "#F2A541"   // 開始・終了（in/out）

    readonly property color histR:     "#FF7A7A"
    readonly property color histG:     "#6EE7A0"
    readonly property color histB:     "#7AB8FF"

    // Noto Sans JP が無い Windows でも崩れないよう Yu Gothic UI を既定にする
    readonly property string font:     "Yu Gothic UI"
    readonly property string mono:     "Consolas"

    readonly property int radius:      12
}
