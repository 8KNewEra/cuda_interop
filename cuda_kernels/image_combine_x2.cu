extern "C"
__global__ void image_combine_x2_kernel(
    uint8_t* out, size_t pitchOut,
    const uint8_t* img1, size_t pitch1,
    const uint8_t* img2, size_t pitch2,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width * 2 || y >= height) return;

    int out_byte_x = x * 4;

    uint8_t* outRow = out + y * pitchOut;

    if (x < width) {
        const uint8_t* row1 = img1 + y * pitch1;
        outRow[out_byte_x + 0] = row1[x * 4 + 0];
        outRow[out_byte_x + 1] = row1[x * 4 + 1];
        outRow[out_byte_x + 2] = row1[x * 4 + 2];
        outRow[out_byte_x + 3] = row1[x * 4 + 3];
    } else {
        const uint8_t* row2 = img2 + y * pitch2;
        int src_x = x - width;
        outRow[out_byte_x + 0] = row2[src_x * 4 + 0];
        outRow[out_byte_x + 1] = row2[src_x * 4 + 1];
        outRow[out_byte_x + 2] = row2[src_x * 4 + 2];
        outRow[out_byte_x + 3] = row2[src_x * 4 + 3];
    }
}
