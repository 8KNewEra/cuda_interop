extern "C"
__global__ void nv12x8_to_rgba_merge_kernel(
    const uint8_t* y0, size_t pitchY0, const uint8_t* uv0, size_t pitchUV0,
    const uint8_t* y1, size_t pitchY1, const uint8_t* uv1, size_t pitchUV1,
    const uint8_t* y2, size_t pitchY2, const uint8_t* uv2, size_t pitchUV2,
    const uint8_t* y3, size_t pitchY3, const uint8_t* uv3, size_t pitchUV3,
    const uint8_t* y4, size_t pitchY4, const uint8_t* uv4, size_t pitchUV4,
    const uint8_t* y5, size_t pitchY5, const uint8_t* uv5, size_t pitchUV5,
    const uint8_t* y6, size_t pitchY6, const uint8_t* uv6, size_t pitchUV6,
    const uint8_t* y7, size_t pitchY7, const uint8_t* uv7, size_t pitchUV7,

    uint8_t* out, size_t pitchOut,

    int outW, int outH,
    int srcW, int srcH
)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH)
        return;

    int quadX = ox / srcW;
    int quadY = oy / srcH;

    int q = quadY * 4 + quadX;

    const uint8_t* Yp;
    const uint8_t* UVp;
    size_t pitchY, pitchUV;

    switch (q) {
    case 0: Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0; break;
    case 1: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1; break;
    case 2: Yp = y2; UVp = uv2; pitchY = pitchY2; pitchUV = pitchUV2; break;
    case 3: Yp = y3; UVp = uv3; pitchY = pitchY3; pitchUV = pitchUV3; break;
    case 4: Yp = y4; UVp = uv4; pitchY = pitchY4; pitchUV = pitchUV4; break;
    case 5: Yp = y5; UVp = uv5; pitchY = pitchY5; pitchUV = pitchUV5; break;
    case 6: Yp = y6; UVp = uv6; pitchY = pitchY6; pitchUV = pitchUV6; break;
    default:Yp = y7; UVp = uv7; pitchY = pitchY7; pitchUV = pitchUV7; break;
    }

    int sx = ox - quadX * srcW;
    int sy = oy - quadY * srcH;

    int Y = (int)Yp[sy * pitchY + sx];

    int uv_x = sx & ~1;
    int uv_y = sy >> 1;

    int U = (int)UVp[uv_y * pitchUV + uv_x + 0] - 128;
    int V = (int)UVp[uv_y * pitchUV + uv_x + 1] - 128;

    int C = Y - 16;
    int R = (298 * C + 409 * V + 128) >> 8;
    int G = (298 * C - 100 * U - 208 * V + 128) >> 8;
    int B = (298 * C + 516 * U + 128) >> 8;

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    ((uchar4*)((uint8_t*)out + oy * pitchOut))[ox] =
        make_uchar4(R, G, B, 255);
}
