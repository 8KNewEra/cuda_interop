extern "C"
__global__ void image_split_x4_kernel(
    uint8_t* Out0, size_t pitch0,
    uint8_t* Out1, size_t pitch1,
    uint8_t* Out2, size_t pitch2,
    uint8_t* Out3, size_t pitch3,
    const uint8_t* In, size_t pitchIn,
    int width, int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint8_t* src0 = In + (y)          * pitchIn + (x)          * 4;
    const uint8_t* src1 = In + (y)          * pitchIn + (x + width)  * 4;
    const uint8_t* src2 = In + (y + height) * pitchIn + (x)          * 4;
    const uint8_t* src3 = In + (y + height) * pitchIn + (x + width)  * 4;

    uint8_t* dst0 = Out0 + y * pitch0 + x * 4;
    uint8_t* dst1 = Out1 + y * pitch1 + x * 4;
    uint8_t* dst2 = Out2 + y * pitch2 + x * 4;
    uint8_t* dst3 = Out3 + y * pitch3 + x * 4;

    *(uint32_t*)dst0 = *(const uint32_t*)src0;
    *(uint32_t*)dst1 = *(const uint32_t*)src1;
    *(uint32_t*)dst2 = *(const uint32_t*)src2;
    *(uint32_t*)dst3 = *(const uint32_t*)src3;
}
