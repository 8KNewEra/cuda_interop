// ==============================================================
// 10bit 経路のカーネルへ値渡しする引数（タイル分割の入出力）
//   .cu（decode_10bit / encode_10bit）と cuda_imageprocess.cpp の両方から読む。
//   注意: 並びを変えたら両方を再ビルドする（qmake の依存で .cu は自動で再コンパイル）
// ==============================================================
#ifndef TILES_10BIT_H
#define TILES_10BIT_H
#include <stddef.h>
#include <stdint.h>

// NVDEC のタイル（デコード入力）。タイルは左上から行ごとの順。8bit の NV12 も同じ並びで渡す（decode_rows.cuh）
struct P010Tiles {
    const uint8_t* y [8];
    const uint8_t* uv[8];
    size_t         yp[8];
    size_t         uvp[8];
    int            cols;    // 横のタイル数（1 / 2 / 4）
    int            count;   // タイル数（1 / 2 / 4 / 8）
    int            srcW;    // 1 タイルの幅・高さ
    int            srcH;
    // 圧縮転送で届いたタイル（docs/DECODE_CORE.md 8.5.2）。packed[q] = 1 なら y / uv は詰めた並び
    //   （pitch = 1 行のバイト数）で、値はそのまま（P010 は >>6 済み）。delta = 1 なら行内の差分（ジグザグ）で、変換と一緒に戻す
    uint8_t        packed[8];
    int            delta;
};

// NVENC へ渡す NV12 タイル（エンコード出力）。タイルは左上から行ごとの順。10bit の P010 も同じ並びで渡す
struct NV12Tiles {
    uint8_t* y [8];
    uint8_t* uv[8];
    size_t   yp[8];
    size_t   uvp[8];
    int      cols;      // 横のタイル数
    int      count;     // タイル数
    int      outW;      // 1 タイルの幅・高さ
    int      outH;
    // 圧縮転送で送るタイル（encode_rows.cuh。docs/ENCODE_CORE.md 5.1）。packed[q] = 1 なら y / uv は XferCodec の
    //   詰めた並び（pitch = 1 行のバイト数）で、値はそのまま（P010 も <<6 しない）。delta = 1 なら行内の差分（ジグザグ）で書く
    //   （P010Tiles の packed / delta と対）
    uint8_t  packed[8];
    int      delta;
};

#endif // TILES_10BIT_H
