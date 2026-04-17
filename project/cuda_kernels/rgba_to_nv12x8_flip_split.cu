extern "C"
__global__ void rgba_to_nv12x8_flip_split_kernel(
    const uint8_t* In, size_t pitchIn,

    uint8_t* y0, size_t pitchY0, uint8_t* uv0, size_t pitchUV0,
    uint8_t* y1, size_t pitchY1, uint8_t* uv1, size_t pitchUV1,
    uint8_t* y2, size_t pitchY2, uint8_t* uv2, size_t pitchUV2,
    uint8_t* y3, size_t pitchY3, uint8_t* uv3, size_t pitchUV3,
    uint8_t* y4, size_t pitchY4, uint8_t* uv4, size_t pitchUV4,
    uint8_t* y5, size_t pitchY5, uint8_t* uv5, size_t pitchUV5,
    uint8_t* y6, size_t pitchY6, uint8_t* uv6, size_t pitchUV6,
    uint8_t* y7, size_t pitchY7, uint8_t* uv7, size_t pitchUV7,

    int srcW, int srcH,
    int outW, int outH
)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= srcW || sy >= srcH)
        return;

    int fy = srcH - 1 - sy;

    uchar4 rgba = ((uchar4*)((uint8_t*)In + fy * pitchIn))[sx];

    float R = rgba.x;
    float G = rgba.y;
    float B = rgba.z;

    float yf = 0.257f * R + 0.504f * G + 0.098f * B + 16.0f;
    float uf = -0.148f * R - 0.291f * G + 0.439f * B + 128.0f;
    float vf = 0.439f * R - 0.368f * G - 0.071f * B + 128.0f;

    uint8_t Y = (uint8_t)fminf(fmaxf(yf, 0.0f), 255.0f);
    uint8_t U = (uint8_t)fminf(fmaxf(uf, 0.0f), 255.0f);
    uint8_t V = (uint8_t)fminf(fmaxf(vf, 0.0f), 255.0f);

    int quadX = sx / outW;
    int quadY = sy / outH;

    if (quadX >= 4 || quadY >= 2)
        return;

    int q = quadY * 4 + quadX; 

    uint8_t* Yp;
    uint8_t* UVp;
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

    int dx = sx - quadX * outW;
    int dy = sy - quadY * outH;

    Yp[dy * pitchY + dx] = (uint8_t)Y;

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        int uv_x = dx;
        int uv_y = dy >> 1;
        uint8_t* uv = UVp + uv_y * pitchUV + uv_x;
        uv[0] = (uint8_t)U;
        uv[1] = (uint8_t)V;
    }
}
