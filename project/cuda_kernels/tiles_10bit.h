// ==============================================================
// 10bit 経路のカーネルへ値渡しする引数（タイル分割の入出力）
//   .cu（decode_10bit / encode_10bit）と cuda_imageprocess.cpp の両方から読む。
//   注意: 並びを変えたら両方を再ビルドする（qmake の依存で .cu は自動で再コンパイル）
// ==============================================================
#ifndef TILES_10BIT_H
#define TILES_10BIT_H
#include <stddef.h>
#include <stdint.h>

// NVDEC の P010 タイル（デコード入力）。タイルは左上から行ごとの順
struct P010Tiles {
    const uint8_t* y [8];
    const uint8_t* uv[8];
    size_t         yp[8];
    size_t         uvp[8];
    int            cols;    // 横のタイル数（1 / 2 / 4）
    int            count;   // タイル数（1 / 2 / 4 / 8）
    int            srcW;    // 1 タイルの幅・高さ
    int            srcH;
    // 圧縮転送で届いたタイル（docs/MULTI_GPU_DECODE.md）。packed[q] = 1 なら y / uv は詰めた並び
    //   （pitch = 1 行のバイト数）で、値は 10bit（>>6 済み）。delta = 1 なら行内の差分（ジグザグ）で、変換と一緒に戻す
    uint8_t        packed[8];
    int            delta;
};

// NVENC へ渡す NV12 タイル（エンコード出力）。タイルは左上から行ごとの順
struct NV12Tiles {
    uint8_t* y [8];
    uint8_t* uv[8];
    size_t   yp[8];
    size_t   uvp[8];
    int      cols;      // 横のタイル数
    int      count;     // タイル数
    int      outW;      // 1 タイルの幅・高さ
    int      outH;
};

#endif // TILES_10BIT_H
