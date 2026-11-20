import QtQuick
import CudaViewer

// 推移グラフ（右端が最新。動画情報・性能 HUD・エンコード設定で共用）
//   中身は C++ の SparklineItem（src/qmlui/sparklineitem.*。シーングラフの三角形で描く）。
//   以前の Canvas 版は描き直すたびに GUI スレッドで JS とソフトウェア描画を回し、映像の upload と swap の間に
//   割り込んで表示の間隔を乱していたので置き換えた（docs/IMG_PROCESS_CORE.md 13.1）
//   記録は呼び出し側が持ち、新しい配列を渡す（capacity = 横軸の点数）
//   values: 古い → 新しい順の配列。null は欠け（線を切る）
//   maxValue: 縦軸の上限、refValue >= 0 なら基準線（破線）を引く
SparklineItem {
    capacity: 60
    maxValue: 100
    refValue: -1
    lineColor: Theme.accent
    refColor: Theme.muted2
    bgColor: Theme.border
    bgOpacity: 0.45
    fillOpacity: 0.18
    lineWidth: 1.5
    clip: true            // 線の縁取り（1px）が枠の外へはみ出さないように
}
