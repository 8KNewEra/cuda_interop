#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2 texelSize;
uniform int u_filterEnabled;

void main()
{
    // フィルタ無効時はそのまま出力
    if (u_filterEnabled == 0) {
        FragColor = texture(tex, TexCoord);
        return;
    }

    // Sobel カーネル（ivec3 で定義）
    const ivec3 kernelX[3] = ivec3[](
        ivec3(-1, 0, 1),
        ivec3(-2, 0, 2),
        ivec3(-1, 0, 1)
    );

    const ivec3 kernelY[3] = ivec3[](
        ivec3(-1, -2, -1),
        ivec3( 0,  0,  0),
        ivec3( 1,  2,  1)
    );

    float gx = 0.0;
    float gy = 0.0;

    // 3x3 サンプリング（ループはGPU側で展開される）
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            vec2 offset = vec2(i - 1, j - 1) * texelSize;
            float gray = texture(tex, TexCoord + offset).r;

            gx += float(kernelX[j][i]) * gray;
            gy += float(kernelY[j][i]) * gray;
        }
    }

    float magnitude = sqrt(gx * gx + gy * gy);
    FragColor = vec4(vec3(magnitude), 1.0);
}
