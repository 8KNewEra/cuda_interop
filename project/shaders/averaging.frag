#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2 texelSize;
uniform int u_filterEnabled;

// 係数を uniform で受ける（QML 版）。u_radius = 0 なら下の固定 7x7 を使う（Widgets 版）
//   u_kernel は行優先 (2r+1)×(2r+1)、正規化済み。最大 9×9
uniform int   u_radius;
uniform float u_kernel[81];

void main()
{
    if (u_filterEnabled == 0) {
        FragColor = texture(tex, TexCoord);
        return;
    }

    if (u_radius > 0) {
        int n = 2 * u_radius + 1;
        vec3 acc = vec3(0.0);
        for (int j = -u_radius; j <= u_radius; j++) {
            for (int i = -u_radius; i <= u_radius; i++) {
                vec3 c = texture(tex, TexCoord + texelSize * vec2(i, j)).rgb;
                acc += c * u_kernel[(j + u_radius) * n + (i + u_radius)];
            }
        }
        FragColor = vec4(acc, 1.0);
        return;
    }

    // 7x7 平均化フィルタ（全て 1/49）
    const float w = 1.0 / 49.0;

    vec3 sum = vec3(0.0);

    // 7x7 box filter
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {

            vec3 c = texture(tex, TexCoord + texelSize * vec2(i, j)).rgb;
            sum += c * w;
        }
    }

    FragColor = vec4(sum, 1.0);
}
