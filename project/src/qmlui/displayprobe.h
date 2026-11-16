#ifndef DISPLAYPROBE_H
#define DISPLAYPROBE_H

// ==============================================================
// 表示経路の色深度を調べる（動画情報の Process/Render Depth の「→ 8bit」判定用）
//
//   画面に出るまでの 3 段のうち、アプリの外の 2 段を Windows（DXGI）から読んで PlayerController に渡す
//     画面合成  : Windows（DWM）が合成に使っている形式。DuplicateOutput1 の形式で分かる
//                 通常の SDR デスクトップ = 8bit、HDR / 自動カラー管理（ACM）が有効 = FP16（16bit）
//                 → setCompositionBits(8 / 10 / 16。0 = 取れない)
//     合成を通るか: 全画面にしたときだけ調べる。描き直しても Desktop Duplication に新しいフレームが
//                 来なければ、合成を通らずに直接モニターへ出している（独立フリップ）
//                 → setComposed(1 = 通る / 0 = 通らない / -1 = 不明)。ウィンドウ表示は常に 1
//     モニター出力: DXGI_OUTPUT_DESC1::BitsPerColor（NVIDIA の 8bpc / 10bpc 設定）→ setOutputBits
//     NVIDIA Deep color: ドライバ設定「Deep color for 3D applications」（OGL_DEEP_COLOR_SCANOUT。NVAPI の DRS で読む）
//                 有効なら OpenGL の 10bit ウィンドウはドライバが合成の外で 10bit のまま出す
//                 （実機 RTX 5000 Ada・SDR・ウィンドウ表示で、8bit 起動との見比べで確認済み）→ setDeepColorScanout
//
//   調べるのは 起動時・別モニターへの移動・アプリに戻ったとき・全画面の切り替え のときだけ
//   限界: Deep color は「設定が有効か」までしか分からない（そのウィンドウが実際にその経路で出ているかは
//         公開 API では取れない）。設定が有効で色バッファが 10bit なら、その経路で出ているとみなす
// ==============================================================

class QQuickWindow;
class PlayerController;

namespace DisplayProbe {
void install(QQuickWindow* window, PlayerController* player);
}

#endif // DISPLAYPROBE_H
