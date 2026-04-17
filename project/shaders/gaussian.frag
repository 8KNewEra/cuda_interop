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
