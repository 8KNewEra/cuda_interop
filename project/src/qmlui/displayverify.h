#ifndef DISPLAYVERIFY_H
#define DISPLAYVERIFY_H

// ==============================================================
// 表示の 10bit 検証（検証用ビルドだけ。CONFIG+=display_verify で有効）
//
//   Ctrl+Shift+D で、マウスカーソルの高さの 1 行（映像の表示枠の範囲）を 2 か所で読み取り、
//   値の段数をログに出す。リリース・通常のビルドには含まれない（.pro の display_verify ブロック）
//
//   ② 色バッファ: シーングラフが描き終えた直後（swap 前）に glReadPixels で読み戻す
//                 → アプリが本当に 10bit で描いているか
//   ③ 画面合成  : Desktop Duplication（DuplicateOutput1、FP16）で合成後の画面を読む
//                 → Windows（DWM）が合成で 8bit に落としていないか
//
//   判定に使う値
//     段数     : 行の中の異なる値の数。test ramp10.avi（docs 参照）の上半分なら 10bit で約 1024、8bit で約 256
//     8bit 外  : ②のみ。8bit の格子（v8 → 10bit）に乗らない値の割合。0% なら 8bit 相当
//     平均幅   : 同じ値が続く画素数の平均（拡大率 1 の ramp10.avi なら 10bit 4px / 8bit 16px）
//   ③は FP16 で線形化されている（ガンマが違う）ので段数と幅だけ見る（単調な変換なので段数は変わらない）
// ==============================================================

class QQuickWindow;

namespace DisplayVerify {
void install(QQuickWindow* window);
}

#endif // DISPLAYVERIFY_H
