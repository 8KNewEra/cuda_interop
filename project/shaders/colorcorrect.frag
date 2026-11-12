#version 450 core

// 画像補正 ＋ 色補正 ＋ トーンカーブ（1 パスにまとめる）
//   画像補正：明るさ → コントラスト → シャドウ / ハイライト → ガンマ
//   色補正  ：色温度・ティント → R/G/B ゲイン・オフセット → R/G/B ガンマ → 彩度
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec2  texelSize;        // 使わない（他のシェーダと uniform をそろえるため）
uniform int   u_filterEnabled;

// ---- 画像補正 ----
uniform int   u_toneOn;
uniform float u_brightness;     // 足す量（-0.5..0.5）
uniform float u_contrast;       // 0.5 を中心に掛ける倍率（0..2）
uniform float u_shadows;        // -1..1（＋で暗部を持ち上げ、－で沈める）
uniform float u_highlights;     // -1..1（－で明部を抑え、＋で持ち上げる）
uniform float u_gamma;          // 1.0 = そのまま。大きいほど中間調が明るい

// ---- 色補正 ----
uniform int   u_rgbOn;
uniform vec3  u_wb;             // 色温度・ティントから作った R/G/B 倍率（明るさは保つ）
uniform vec3  u_gain;           // R/G/B の倍率（0..2）
uniform vec3  u_offset;         // R/G/B に足す量（-0.5..0.5）
uniform vec3  u_rgbGamma;       // R/G/B のガンマ（1.0 = そのまま）
uniform float u_saturation;     // 0 = 白黒、1 = そのまま、2 = 2 倍

// ---- トーンカーブ ----
//   256 段の対応表（R/G/B。RGB 全体のカーブを先に掛けた合成済み）。線形補間で引く
uniform int       u_curveOn;
uniform sampler1D u_curve;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);   // Rec.709

void main()
{
    vec3 c = texture(tex, TexCoord).rgb;
    if (u_filterEnabled == 0) {
        FragColor = vec4(c, 1.0);
        return;
    }

    if (u_toneOn != 0) {
        c = c + u_brightness;
        c = (c - 0.5) * u_contrast + 0.5;
        c = clamp(c, 0.0, 1.0);

        // シャドウ / ハイライト：輝度で暗部・明部を分け、その部分だけ明るさを足し引きする
        //   暗部の重み：輝度 0 で 1 → 0.5 で 0、明部の重み：0.5 で 0 → 1 で 1（中間調はほぼ動かない）
        //   足す量は輝度の差分を RGB に同じだけ加える（色味を崩しにくい）
        float y  = dot(c, kLuma);
        float ws = 1.0 - smoothstep(0.0, 0.5, y);
        float wh = smoothstep(0.5, 1.0, y);
        c = clamp(c + (u_shadows * ws + u_highlights * wh) * 0.35, 0.0, 1.0);

        c = pow(c, vec3(1.0 / max(u_gamma, 0.01)));
    }

    if (u_rgbOn != 0) {
        c = clamp(c * u_wb * u_gain + u_offset, 0.0, 1.0);
        c = pow(c, 1.0 / max(u_rgbGamma, vec3(0.01)));
        c = mix(vec3(dot(c, kLuma)), c, u_saturation);
    }

    if (u_curveOn != 0) {
        c = clamp(c, 0.0, 1.0);
        // テクセルの中心（0.5/256 .. 255.5/256）に合わせて読む
        vec3 u = (c * 255.0 + 0.5) / 256.0;
        c = vec3(texture(u_curve, u.r).r, texture(u_curve, u.g).g, texture(u_curve, u.b).b);
    }

    FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
