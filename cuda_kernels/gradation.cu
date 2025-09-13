extern "C"
__global__ void gradetion_kernel(
    uchar3* output_bgr, int output_bgr_step,
    const uchar3* input_bgr, int input_bgr_step,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uchar3* input_row  = (const uchar3*)((const char*)input_bgr  + y * input_bgr_step);
    uchar3*       output_row = (uchar3*)((char*)output_bgr + y * output_bgr_step);

    uchar3 in_pixel = input_row[x];

    float grad = (float)x / (width - 1);

    float alpha = 0.5f + grad * 2.0f;

    uchar3 out_pixel;
    out_pixel.x = min(255, max(0, (int)((in_pixel.x - 128) * alpha + 128)));
    out_pixel.y = min(255, max(0, (int)((in_pixel.y - 128) * alpha + 128)));
    out_pixel.z = min(255, max(0, (int)((in_pixel.z - 128) * alpha + 128)));

    output_row[x] = out_pixel;
}
