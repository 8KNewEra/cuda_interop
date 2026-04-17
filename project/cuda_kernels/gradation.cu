extern "C"
__global__ void gradetion_kernel(
    uint8_t* output_rgba, int output_rgba_step,
    const uint8_t* input_rgba, int input_rgba_step,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uchar4* input_row  = (const uchar4*)((const char*)input_rgba  + y * input_rgba_step);
    uchar4*       output_row = (uchar4*)((char*)output_rgba + y * output_rgba_step);

    uchar4 in_pixel = input_row[x];

    float grad = (float)x / (width - 1);
    float alpha = 0.5f + grad * 2.0f;

    uchar4 out_pixel;
    out_pixel.x = min(255, max(0, (int)((in_pixel.x - 128) * alpha + 128))); // R
    out_pixel.y = min(255, max(0, (int)((in_pixel.y - 128) * alpha + 128))); // G
    out_pixel.z = min(255, max(0, (int)((in_pixel.z - 128) * alpha + 128))); // B
    out_pixel.w = in_pixel.w;

    output_row[x] = out_pixel;
}

