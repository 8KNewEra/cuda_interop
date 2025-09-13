#version 330

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2 texelSize;
uniform int u_filterEnabled;  // 追加

void main() {
    if (u_filterEnabled == 0) {
        FragColor = texture(tex, TexCoord);  // そのまま表示
        return;
    }

    // Sobelフィルタ適用部分（例）
    float kernelX[9] = float[](
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );

    float kernelY[9] = float[](
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    );

    float gx = 0.0;
    float gy = 0.0;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * texelSize;
            float gray = texture(tex, TexCoord + offset).r;
            int idx = (y + 1) * 3 + (x + 1);
            gx += kernelX[idx] * gray;
            gy += kernelY[idx] * gray;
        }
    }

    float magnitude = sqrt(gx * gx + gy * gy);
    FragColor = vec4(vec3(magnitude), 1.0);
}
