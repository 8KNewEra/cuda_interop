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

    float4* out, size_t pitchOut,

    int outW, int outH,
    int srcW, int srcH
)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

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

    float Y = float(Yp[sy * pitchY + sx]);
    int uv_x = sx & ~1;
    int uv_y = sy >> 1;

    float U = float(UVp[uv_y * pitchUV + uv_x + 0]);
    float V = float(UVp[uv_y * pitchUV + uv_x + 1]);

    float Yf = (Y - 16.0f)  / 219.0f;
    float Uf = (U - 128.0f) / 224.0f;
    float Vf = (V - 128.0f) / 224.0f;

    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);

    float R = Yf + 1.5748f * Vf;
    float G = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float B = Yf + 1.8556f * Uf;

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    float4* row = (float4*)((uint8_t*)out + oy * pitchOut);
    row[ox] = make_float4(R, G, B, 1.0f);
}

