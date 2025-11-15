#version 450 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2 texelSize;
uniform int u_filterEnabled;

void main()
{
    if (u_filterEnabled == 0) {
        FragColor = texture(tex, TexCoord);
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
