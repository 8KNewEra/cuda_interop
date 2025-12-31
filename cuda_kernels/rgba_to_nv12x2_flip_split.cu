extern "C"
__global__ void rgba_to_nv12x2_flip_split_kernel(
    const uint8_t* In, size_t pitchIn,

    uint8_t* y0,  size_t pitchY0,
    uint8_t* uv0, size_t pitchUV0,

    uint8_t* y1,  size_t pitchY1,
    uint8_t* uv1, size_t pitchUV1,

    int srcW, int srcH,
    int outW, int outH
)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= srcW || sy >= srcH)
        return;

    int fy = srcH - 1 - sy;

    uchar4 rgba =
        ((uchar4*)((uint8_t*)In + fy * pitchIn))[sx];

    int R = rgba.x;
    int G = rgba.y;
    int B = rgba.z;

    int Y = ( 66 * R + 129 * G +  25 * B + 128) >> 8;
    int U = (-38 * R -  74 * G + 112 * B + 128) >> 8;
    int V = (112 * R -  94 * G -  18 * B + 128) >> 8;

    Y = min(max(Y + 16,  0), 255);
    U = min(max(U + 128, 0), 255);
    V = min(max(V + 128, 0), 255);

    int quadX = (sx >= outW);

    uint8_t* Yp;
    uint8_t* UVp;
    size_t pitchY;
    size_t pitchUV;

    if (quadX == 0) {
        Yp = y0; UVp = uv0;
        pitchY = pitchY0; pitchUV = pitchUV0;
    } else {
        Yp = y1; UVp = uv1;
        pitchY = pitchY1; pitchUV = pitchUV1;
    }

    int dx = sx - quadX * outW;
    int dy = sy;

    Yp[dy * pitchY + dx] = (uint8_t)Y;

    if ((dx & 1) == 0 && (dy & 1) == 0) {
        int uv_x = dx;
        int uv_y = dy >> 1;
        uint8_t* uv = UVp + uv_y * pitchUV + uv_x;
        uv[0] = (uint8_t)U;
        uv[1] = (uint8_t)V;
    }
}

