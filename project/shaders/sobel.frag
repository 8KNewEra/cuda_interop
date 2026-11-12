#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2 texelSize;
uniform int u_filterEnabled;

// u_mode: 0 = Widgets 版の従来動作（R を明るさとしてエッジを白黒で出す）
//         1 = エッジのみ（輝度で計算）
//         2 = ピーキング（元映像に、しきい値を超えた輪郭だけ色を重ねる）
uniform int   u_mode;
uniform float u_threshold;   // ピーキングのしきい値（0..1）
uniform vec3  u_peakColor;

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

    vec3  center = vec3(0.0);
    float gx = 0.0;
    float gy = 0.0;

    // 3x3 サンプリング
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            vec2 offset = vec2(i - 1, j - 1) * texelSize;
            vec3  rgb  = texture(tex, TexCoord + offset).rgb;
            float gray = (u_mode == 0) ? rgb.r : dot(rgb, vec3(0.299, 0.587, 0.114));
            if (i == 1 && j == 1) center = rgb;   // ピーキングで使う元の画素（追加の読み込みなし）

            gx += float(kernelX[j][i]) * gray;
            gy += float(kernelY[j][i]) * gray;
        }
    }

    float magnitude = sqrt(gx * gx + gy * gy);

    if (u_mode == 2) {
        // しきい値を超えた分に応じて色を濃くする（ちょうど超えたところは薄め）
        float t = clamp((magnitude - u_threshold) * 4.0 + 0.35, 0.0, 1.0);
        vec3  o = (magnitude > u_threshold) ? mix(center, u_peakColor, t) : center;
        FragColor = vec4(o, 1.0);
        return;
    }
    FragColor = vec4(vec3(magnitude), 1.0);
}
