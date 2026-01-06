extern "C"
__global__ void nv12x2_to_rgba_merge_kernel(
    const uint8_t* y0,  size_t pitchY0,
    const uint8_t* uv0, size_t pitchUV0,

    const uint8_t* y1,  size_t pitchY1,
    const uint8_t* uv1, size_t pitchUV1,

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
    default: Yp = y1; UVp = uv1; pitchY = pitchY1; pitchUV = pitchUV1;
            sx = ox - srcW;    sy = oy;           break;
    }
    
    int Y  = (int)Yp[sy * pitchY + sx];

    int uv_x = sx & ~1;
    int uv_y = sy >> 1;

    int U = (int)UVp[uv_y * pitchUV + uv_x + 0];
    int V = (int)UVp[uv_y * pitchUV + uv_x + 1];

    float fY = (float)Y - 16.0f;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = 1.164f * fY + 1.596f * fV;
    float G = 1.164f * fY - 0.392f * fU - 0.813f * fV;
    float B = 1.164f * fY + 2.017f * fU;

    uint8_t r = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
    uint8_t g = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    uint8_t b = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);

    uchar4 pixel;
    pixel.x = r;
    pixel.y = g;
    pixel.z = b;
    pixel.w = 255;

    uchar4* row = (uchar4*)((uint8_t*)out + oy * pitchOut);
    row[ox] = pixel;
}
