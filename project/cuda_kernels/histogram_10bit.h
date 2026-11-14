// ==============================================================
// 10bit 経路のヒストグラムの集計先（R / G / B 各 1024 段 = 10bit の値ごと）
//   histogram_10bit.cu と cuda_imageprocess.cpp / VideoEngine の両方から読む。
//   統計の結果は 8bit 経路と同じ HistStats（__global__.h）に入れる（値は 0..1023）
// ==============================================================
#ifndef HISTOGRAM_10BIT_H
#define HISTOGRAM_10BIT_H

#define HIST10_BINS 1024

struct HistData10 {
    unsigned int hist_r[HIST10_BINS];
    unsigned int hist_g[HIST10_BINS];
    unsigned int hist_b[HIST10_BINS];
};

#endif // HISTOGRAM_10BIT_H
