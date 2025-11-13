struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
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
    const HistData* Histdata,
    HistStats* out_stats)
{
    int x = threadIdx.x;

    __shared__ int s_min_r[256], s_max_r[256];
    __shared__ int s_min_g[256], s_max_g[256];
    __shared__ int s_min_b[256], s_max_b[256];

    __shared__ unsigned long long s_sum_r[256], s_count_r[256];
    __shared__ unsigned long long s_sum_g[256], s_count_g[256];
    __shared__ unsigned long long s_sum_b[256], s_count_b[256];

    unsigned int r = Histdata->hist_r[x];
    unsigned int g = Histdata->hist_g[x];
    unsigned int b = Histdata->hist_b[x];

    s_min_r[x] = (r > 0) ? x : 9999;
    s_min_g[x] = (g > 0) ? x : 9999;
    s_min_b[x] = (b > 0) ? x : 9999;

    s_max_r[x] = (r > 0) ? x : -1;
    s_max_g[x] = (g > 0) ? x : -1;
    s_max_b[x] = (b > 0) ? x : -1;

    s_sum_r[x]   = (unsigned long long)x * r;
    s_sum_g[x]   = (unsigned long long)x * g;
    s_sum_b[x]   = (unsigned long long)x * b;

    s_count_r[x] = r;
    s_count_g[x] = g;
    s_count_b[x] = b;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (x < stride) {
            s_min_r[x] = min(s_min_r[x], s_min_r[x + stride]);
            s_min_g[x] = min(s_min_g[x], s_min_g[x + stride]);
            s_min_b[x] = min(s_min_b[x], s_min_b[x + stride]);

            s_max_r[x] = max(s_max_r[x], s_max_r[x + stride]);
            s_max_g[x] = max(s_max_g[x], s_max_g[x + stride]);
            s_max_b[x] = max(s_max_b[x], s_max_b[x + stride]);

            s_sum_r[x] += s_sum_r[x + stride];
            s_sum_g[x] += s_sum_g[x + stride];
            s_sum_b[x] += s_sum_b[x + stride];

            s_count_r[x] += s_count_r[x + stride];
            s_count_g[x] += s_count_g[x + stride];
            s_count_b[x] += s_count_b[x + stride];
        }
        __syncthreads();
    }

    if (x == 0) {
        HistStats out;

        out.min_r = (s_min_r[0] == 9999) ? -1 : s_min_r[0];
        out.min_g = (s_min_g[0] == 9999) ? -1 : s_min_g[0];
        out.min_b = (s_min_b[0] == 9999) ? -1 : s_min_b[0];

        out.max_r = s_max_r[0];
        out.max_g = s_max_g[0];
        out.max_b = s_max_b[0];

        out.avg_r = (s_count_r[0] > 0)
            ? double(s_sum_r[0]) / double(s_count_r[0]) : 0.0;

        out.avg_g = (s_count_g[0] > 0)
            ? double(s_sum_g[0]) / double(s_count_g[0]) : 0.0;

        out.avg_b = (s_count_b[0] > 0)
            ? double(s_sum_b[0]) / double(s_count_b[0]) : 0.0;

        out.avg_r = round(out.avg_r * 10.0) / 10.0;
        out.avg_g = round(out.avg_g * 10.0) / 10.0;
        out.avg_b = round(out.avg_b * 10.0) / 10.0;

        out.max_y_axis = max(Histdata->max_r,max(Histdata->max_g, Histdata->max_b));

        *out_stats = out;
    }
}
