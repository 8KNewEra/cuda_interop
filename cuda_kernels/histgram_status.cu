#include <cuda_fp16.h>

struct HistData {
    unsigned int hist_r[1024];
    unsigned int hist_g[1024];
    unsigned int hist_b[1024];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

struct HistStats {
    int min_r, max_r;
    int min_g, max_g;
    int min_b, max_b;
    double avg_r, avg_g, avg_b;
    int max_y_axis;
};

extern "C"
__global__
void histgram_status_kernel(
    const HistData* hist,
    HistStats* out)
{
    int x = threadIdx.x;
    if (x >= 1024) return;

    __shared__ int s_work_r[1024];
    __shared__ int s_work_g[1024];
    __shared__ int s_work_b[1024];

    __shared__ unsigned long long s_sum_r[1024];
    __shared__ unsigned long long s_sum_g[1024];
    __shared__ unsigned long long s_sum_b[1024];

    unsigned int hr = hist->hist_r[x];
    unsigned int hg = hist->hist_g[x];
    unsigned int hb = hist->hist_b[x];

    s_work_r[x] = (hr > 0) ? x : 1024;
    s_work_g[x] = (hg > 0) ? x : 1024;
    s_work_b[x] = (hb > 0) ? x : 1024;

    __syncthreads();

    for (int stride = 512; stride > 0; stride >>= 1) {
        if (x < stride) {
            s_work_r[x] = min(s_work_r[x], s_work_r[x + stride]);
            s_work_g[x] = min(s_work_g[x], s_work_g[x + stride]);
            s_work_b[x] = min(s_work_b[x], s_work_b[x + stride]);
        }
        __syncthreads();
    }

    int min_r = s_work_r[0];
    int min_g = s_work_g[0];
    int min_b = s_work_b[0];

    s_work_r[x] = (hr > 0) ? x : -1;
    s_work_g[x] = (hg > 0) ? x : -1;
    s_work_b[x] = (hb > 0) ? x : -1;

    __syncthreads();

    for (int stride = 512; stride > 0; stride >>= 1) {
        if (x < stride) {
            s_work_r[x] = max(s_work_r[x], s_work_r[x + stride]);
            s_work_g[x] = max(s_work_g[x], s_work_g[x + stride]);
            s_work_b[x] = max(s_work_b[x], s_work_b[x + stride]);
        }
        __syncthreads();
    }

    int max_r = s_work_r[0];
    int max_g = s_work_g[0];
    int max_b = s_work_b[0];

    s_sum_r[x] = (unsigned long long)x * hr;
    s_sum_g[x] = (unsigned long long)x * hg;
    s_sum_b[x] = (unsigned long long)x * hb;

    __syncthreads();

    for (int stride = 512; stride > 0; stride >>= 1) {
        if (x < stride) {
            s_sum_r[x] += s_sum_r[x + stride];
            s_sum_g[x] += s_sum_g[x + stride];
            s_sum_b[x] += s_sum_b[x + stride];
        }
        __syncthreads();
    }

    if (x == 0) {
        HistStats st;

        st.min_r = (min_r == 1024) ? -1 : min_r;
        st.min_g = (min_g == 1024) ? -1 : min_g;
        st.min_b = (min_b == 1024) ? -1 : min_b;

        st.max_r = max_r;
        st.max_g = max_g;
        st.max_b = max_b;

        unsigned long long count_r = 0;
        unsigned long long count_g = 0;
        unsigned long long count_b = 0;

        for (int i = 0; i < 1024; ++i) {
            count_r += hist->hist_r[i];
            count_g += hist->hist_g[i];
            count_b += hist->hist_b[i];
        }

        st.avg_r = (count_r > 0) ? double(s_sum_r[0]) / double(count_r) : 0.0;
        st.avg_g = (count_g > 0) ? double(s_sum_g[0]) / double(count_g) : 0.0;
        st.avg_b = (count_b > 0) ? double(s_sum_b[0]) / double(count_b) : 0.0;

        st.avg_r = round(st.avg_r * 10.0) / 10.0;
        st.avg_g = round(st.avg_g * 10.0) / 10.0;
        st.avg_b = round(st.avg_b * 10.0) / 10.0;

        st.max_y_axis = max(hist->max_r,
                         max(hist->max_g, hist->max_b));

        *out = st;
    }
}
