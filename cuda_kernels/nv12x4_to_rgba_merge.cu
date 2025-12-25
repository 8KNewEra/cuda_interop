extern "C"
__global__ void nv12x4_to_rgba_merge_kernel(
    const uint8_t* y0,  size_t pitchY0,
    const uint8_t* uv0, size_t pitchUV0,

    const uint8_t* y1,  size_t pitchY1,
    const uint8_t* uv1, size_t pitchUV1,

    const uint8_t* y2,  size_t pitchY2,
    const uint8_t* uv2, size_t pitchUV2,

    const uint8_t* y3,  size_t pitchY3,
    const uint8_t* uv3, size_t pitchUV3,

    uint8_t* out, size_t pitchOut,

    int outW, int outH,
    int srcW, int srcH
)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH)
        return;

    const uint8_t *Yp, *UVp;
    size_t pitchY, pitchUV;
    int sx, sy;

    int quadX = (ox >= srcW);
    int quadY = (oy >= srcH);

    int q = quadY * 2 + quadX;

    switch (q) {
    case 0: Yp = y0; UVp = uv0; pitchY = pitchY0; pitchUV = pitchUV0;
            sx = ox;           sy = oy;           break;
    case 1: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1;
            sx = ox - srcW;    sy = oy;           break;
    case 2: Yp = y2; UVp = uv2; pitchY = pitchY2; pitchUV = pitchUV2;
            sx = ox;           sy = oy - srcH;    break;
    default:Yp = y3; UVp = uv3; pitchY = pitchY3; pitchUV = pitchUV3;
            sx = ox - srcW;    sy = oy - srcH;    break;
    }
    
    int Y  = (int)Yp[sy * pitchY + sx];

    int uv_x = sx & ~1;
    int uv_y = sy >> 1;

    int U = (int)UVp[uv_y * pitchUV + uv_x + 0] - 128;
    int V = (int)UVp[uv_y * pitchUV + uv_x + 1] - 128;

    int C = Y - 16;
    int R = (298 * C + 409 * V + 128) >> 8;
    int G = (298 * C - 100 * U - 208 * V + 128) >> 8;
    int B = (298 * C + 516 * U + 128) >> 8;

    R = R < 0 ? 0 : (R > 255 ? 255 : R);
    G = G < 0 ? 0 : (G > 255 ? 255 : G);
    B = B < 0 ? 0 : (B > 255 ? 255 : B);

    ((uchar4*)((uint8_t*)out + oy * pitchOut))[ox] =
        make_uchar4(R, G, B, 255);
}
