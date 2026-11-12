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
// 0 = ぼかし、1 = シャープ化（アンシャープマスク: 元 ＋ 強さ ×（元 − ぼかし））
uniform int   u_mode;
uniform float u_amount;

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
        if (u_mode == 1) {
            vec3 orig = texture(tex, TexCoord).rgb;
            acc = clamp(orig + u_amount * (orig - acc), 0.0, 1.0);
        }
        FragColor = vec4(acc, 1.0);
        return;
    }

    // 7x7 Gaussian kernel (σ = 2)
    float kernel[49] = float[](
        0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067,
        0.00002292, 0.00078634, 0.00655965, 0.01331511, 0.00655965, 0.00078634, 0.00002292,
        0.00019117, 0.00655965, 0.05472157, 0.11110793, 0.05472157, 0.00655965, 0.00019117,
        0.00038771, 0.01331511, 0.11110793, 0.22508352, 0.11110793, 0.01331511, 0.00038771,
        0.00019117, 0.00655965, 0.05472157, 0.11110793, 0.05472157, 0.00655965, 0.00019117,
        0.00002292, 0.00078634, 0.00655965, 0.01331511, 0.00655965, 0.00078634, 0.00002292,
        0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067
    );

    vec3 sum = vec3(0.0);
    int k = 0;

    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec3 c = texture(tex, TexCoord + texelSize * vec2(i, j)).rgb;
            sum += c * kernel[k++];
        }
    }

    FragColor = vec4(sum, 1.0);
}
