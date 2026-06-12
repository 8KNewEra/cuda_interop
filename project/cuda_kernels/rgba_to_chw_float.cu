struct gpuFrame {
    uint8_t* data = nullptr;
    size_t pitch = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
};

extern "C"
__global__ void rgba_to_chw_float_kernel(gpuFrame src_frame, gpuFrame dst_frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_frame.width || y >= dst_frame.height) return;

    float ratio_w = static_cast<float>(dst_frame.width) / src_frame.width;
    float ratio_h = static_cast<float>(dst_frame.height) / src_frame.height;
    float scale = fminf(ratio_w, ratio_h);

    int scaled_w = static_cast<int>(src_frame.width * scale);
    int scaled_h = static_cast<int>(src_frame.height * scale);

    int offset_x = (dst_frame.width - scaled_w) / 2;
    int offset_y = (dst_frame.height - scaled_h) / 2;

    float inv_scale = 1.0f / scale;

    float* dst_plane = reinterpret_cast<float*>(dst_frame.data);
    int plane_size = dst_frame.width * dst_frame.height;
    int pixel_idx = y * dst_frame.width + x;

    bool is_black_bar = (x < offset_x || x >= offset_x + scaled_w ||
                         y < offset_y || y >= offset_y + scaled_h);

    if (is_black_bar) {
        dst_plane[pixel_idx]                  = 0.0f; // R
        dst_plane[pixel_idx + plane_size]     = 0.0f; // G
        dst_plane[pixel_idx + 2 * plane_size] = 0.0f; // B
        return; 
    }

    float dst_x_adj = x - offset_x;
    float dst_y_adj = y - offset_y;

    float src_x = (dst_x_adj + 0.5f) * inv_scale - 0.5f;
    float src_y = (dst_y_adj + 0.5f) * inv_scale - 0.5f;

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = max(0, min(x0 + 1, src_frame.width - 1));
    int y1 = max(0, min(y0 + 1, src_frame.height - 1));
    x0 = max(0, min(x0, src_frame.width - 1));
    y0 = max(0, min(y0, src_frame.height - 1));
    
    float kx = src_x - floorf(src_x);
    float ky = src_y - floorf(src_y);

    uint8_t* row0 = src_frame.data + (y0 * src_frame.pitch);
    uint8_t* row1 = src_frame.data + (y1 * src_frame.pitch);

    // Red (ch=0)
    float pr00 = static_cast<float>(row0[x0 * 4 + 0]) / 255.0f;
    float pr10 = static_cast<float>(row0[x1 * 4 + 0]) / 255.0f;
    float pr01 = static_cast<float>(row1[x0 * 4 + 0]) / 255.0f;
    float pr11 = static_cast<float>(row1[x1 * 4 + 0]) / 255.0f;
    float r = (1.0f - kx) * (1.0f - ky) * pr00 + kx * (1.0f - ky) * pr10 + (1.0f - kx) * ky * pr01 + kx * ky * pr11;

    // Green (ch=1)
    float pg00 = static_cast<float>(row0[x0 * 4 + 1]) / 255.0f;
    float pg10 = static_cast<float>(row0[x1 * 4 + 1]) / 255.0f;
    float pg01 = static_cast<float>(row1[x0 * 4 + 1]) / 255.0f;
    float pg11 = static_cast<float>(row1[x1 * 4 + 1]) / 255.0f;
    float g = (1.0f - kx) * (1.0f - ky) * pg00 + kx * (1.0f - ky) * pg10 + (1.0f - kx) * ky * pg01 + kx * ky * pg11;

    // Blue (ch=2)
    float pb00 = static_cast<float>(row0[x0 * 4 + 2]) / 255.0f;
    float pb10 = static_cast<float>(row0[x1 * 4 + 2]) / 255.0f;
    float pb01 = static_cast<float>(row1[x0 * 4 + 2]) / 255.0f;
    float pb11 = static_cast<float>(row1[x1 * 4 + 2]) / 255.0f;
    float b = (1.0f - kx) * (1.0f - ky) * pb00 + kx * (1.0f - ky) * pb10 + (1.0f - kx) * ky * pb01 + kx * ky * pb11;

    // 5. 書き込み
    dst_plane[pixel_idx]                  = r;
    dst_plane[pixel_idx + plane_size]     = g;
    dst_plane[pixel_idx + 2 * plane_size] = b;
}