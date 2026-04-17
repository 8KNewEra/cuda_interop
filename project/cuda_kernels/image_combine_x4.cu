extern "C"
__global__ void image_combine_x4_kernel(
    uint8_t* out, size_t pitchOut,
    const uint8_t* img1, size_t pitch1,
    const uint8_t* img2, size_t pitch2,
    const uint8_t* img3, size_t pitch3,
    const uint8_t* img4, size_t pitch4,
    int width, int height,
    int blend)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outW = width * 2;
    int outH = height * 2;
    if (x >= outW || y >= outH) return;

    uint8_t* outRow = out + y * pitchOut;
    int out_byte_x = x * 4;

    const uint8_t* srcA = nullptr;
    const uint8_t* srcB = nullptr;
    size_t pitchA = 0, pitchB = 0;
    int ax, ay, bx, by;
    float alpha = 0.0f;
    bool doBlend = false;

    if (x >= width - blend && x < width + blend) {
        doBlend = true;
        alpha = float(x - (width - blend)) / float(blend * 2);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);

        if (y < height) {
            srcA = img1; pitchA = pitch1;
            srcB = img2; pitchB = pitch2;
            ay = y; ax = width - 1;
            by = y; bx = 0;
        } else {
            srcA = img3; pitchA = pitch3;
            srcB = img4; pitchB = pitch4;
            ay = y - height; ax = width - 1;
            by = y - height; bx = 0;
        }
    }

    else if (y >= height - blend && y < height + blend) {
        doBlend = true;
        alpha = float(y - (height - blend)) / float(blend * 2);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);

        if (x < width) {
            srcA = img1; pitchA = pitch1;
            srcB = img3; pitchB = pitch3;
            ax = x; ay = height - 1;
            bx = x; by = 0;
        } else {
            srcA = img2; pitchA = pitch2;
            srcB = img4; pitchB = pitch4;
            ax = x - width; ay = height - 1;
            bx = x - width; by = 0;
        }
    }

    if (doBlend) {
        const uint8_t* rowA = srcA + ay * pitchA + ax * 4;
        const uint8_t* rowB = srcB + by * pitchB + bx * 4;

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = rowA[c] * (1.0f - alpha) + rowB[c] * alpha;
            outRow[out_byte_x + c] = (uint8_t)(v + 0.5f);
        }
        return;
    }

    const uint8_t* src;
    size_t pitch;
    int sx, sy;

    if (y < height) {
        if (x < width) {
            src = img1; pitch = pitch1; sx = x; sy = y;
        } else {
            src = img2; pitch = pitch2; sx = x - width; sy = y;
        }
    } else {
        if (x < width) {
            src = img3; pitch = pitch3; sx = x; sy = y - height;
        } else {
            src = img4; pitch = pitch4; sx = x - width; sy = y - height;
        }
    }

    const uint8_t* srcRow = src + sy * pitch + sx * 4;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        outRow[out_byte_x + c] = srcRow[c];
    }
}

