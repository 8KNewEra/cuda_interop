struct gpuFrame {
    uint8_t* data = nullptr;
    size_t pitch = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
};

extern "C"
__global__ void chw_float_to_rgba_kernel(gpuFrame src_frame, gpuFrame dst_frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_frame.width || y >= dst_frame.height) return;

    float ratio_w = static_cast<float>(src_frame.width) / dst_frame.width;
    float ratio_h = static_cast<float>(src_frame.height) / dst_frame.height;
    float scale = fminf(ratio_w, ratio_h);

    int scaled_w = static_cast<int>(dst_frame.width * scale);
    int scaled_h = static_cast<int>(dst_frame.height * scale);

    float offset_x = (src_frame.width - scaled_w) / 2.0f;
    float offset_y = (src_frame.height - scaled_h) / 2.0f;

    float src_x = offset_x + (x + 0.5f) * scale - 0.5f;
    float src_y = offset_y + (y + 0.5f) * scale - 0.5f;

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = max(0, min(x0 + 1, src_frame.width - 1));
    int y1 = max(0, min(y0 + 1, src_frame.height - 1));
    x0 = max(0, min(x0, src_frame.width - 1));
    y0 = max(0, min(y0, src_frame.height - 1));

    float kx = src_x - floorf(src_x);
    float ky = src_y - floorf(src_y);

    const float* src_plane = reinterpret_cast<const float*>(src_frame.data);
    int plane_size = src_frame.width * src_frame.height;

    int idx00 = y0 * src_frame.width + x0;
    int idx10 = y0 * src_frame.width + x1;
    int idx01 = y1 * src_frame.width + x0;
    int idx11 = y1 * src_frame.width + x1;

    float pr00 = src_plane[idx00];
    float pr10 = src_plane[idx10];
    float pr01 = src_frame.data ? src_plane[idx01] : 0.0f;
    float pr11 = src_plane[idx11];
    float r_f = (1.0f - kx) * (1.0f - ky) * pr00 + kx * (1.0f - ky) * pr10 + (1.0f - kx) * ky * pr01 + kx * ky * pr11;

    float pg00 = src_plane[plane_size + idx00];
    float pg10 = src_plane[plane_size + idx10];
    float pg01 = src_plane[plane_size + idx01];
    float pg11 = src_plane[plane_size + idx11];
    float g_f = (1.0f - kx) * (1.0f - ky) * pg00 + kx * (1.0f - ky) * pg10 + (1.0f - kx) * ky * pg01 + kx * ky * pg11;

    float pb00 = src_plane[2 * plane_size + idx00];
    float pb10 = src_plane[2 * plane_size + idx10];
    float pb01 = src_plane[2 * plane_size + idx01];
    float pb11 = src_plane[2 * plane_size + idx11];
    float b_f = (1.0f - kx) * (1.0f - ky) * pb00 + kx * (1.0f - ky) * pb10 + (1.0f - kx) * ky * pb01 + kx * ky * pb11;

    uint8_t r = static_cast<uint8_t>(__saturatef(r_f) * 255.0f + 0.5f);
    uint8_t g = static_cast<uint8_t>(__saturatef(g_f) * 255.0f + 0.5f);
    uint8_t b = static_cast<uint8_t>(__saturatef(b_f) * 255.0f + 0.5f);

    uint8_t* dst_row = dst_frame.data + (y * dst_frame.pitch);
    int dst_idx = x * 4; 

    dst_row[dst_idx + 0] = r;
    dst_row[dst_idx + 1] = g;
    dst_row[dst_idx + 2] = b;
    dst_row[dst_idx + 3] = 255;
}

