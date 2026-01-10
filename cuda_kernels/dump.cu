#include <stdio.h>

extern "C"
__global__ void dump_kernel(uint8_t* p)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf(
            "GPU data: %02x %02x %02x %02x %02x %02x %02x %02x\n",
            p[0], p[1], p[2], p[3],
            p[4], p[5], p[6], p[7]
        );
    }
}

