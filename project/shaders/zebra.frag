#version 450 core

// 表示専用の最終パス（fbo を読んで tempTexture に描く。fbo 自体は変えない）
//   適用順:
//   1) 比較表示   : 境目の片側に補正前（CUDA からの入力）を出し、境目に線を引く
//   2) チャンネル : R / G / B / 輝度 のどれか 1 つを白黒で出す
//   3) ゼブラ     : 輝度（Rec.709、チャンネル表示前の色）が u_high 以上 → 黒の斜め縞、
//                   u_low 以下 → 青の斜め縞
//   4) タイル境界 : 境界をまたぐ差が両側の内側の差より不自然に大きい所を色で強調
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D tex;
uniform int   u_zebraOn;
uniform int   u_highOn;
uniform float u_high;     // 0..1
uniform int   u_lowOn;
uniform float u_low;      // 0..1
uniform float u_period;   // 縞 1 周期の幅（映像のピクセル）
uniform int   u_channel;  // -1 = そのまま、0 = R、1 = G、2 = B、3 = 輝度

uniform sampler2D u_orig;      // 補正前（CUDA からの入力。上下が逆なので v を反転して読む）
uniform int   u_compareOn;
uniform int   u_compareDir;    // 0 = 左右（左が補正前）、1 = 上下（上が補正前）
uniform float u_comparePos;    // 境目の位置 0..1（左端 / 上端から）
uniform float u_lineW;         // 境目の線の太さ（映像のピクセル）

uniform int   u_seamOn;
uniform ivec2 u_tileGrid;      // タイルの列数・行数
uniform ivec2 u_tileSize;      // 1 タイルの幅・高さ（ピクセル）
uniform float u_seamThr;       // 段差のしきい値（0..1）
uniform int   u_seamBand;      // 強調する帯の半幅（ピクセル）
uniform vec3  u_seamColor;     // 強調の色

// 補正前（入力そのまま）を上から数えた座標で読む。フィルタの影響を受けずに判定する
vec3 origAt(ivec2 p)
{
    ivec2 sz = textureSize(u_orig, 0);
    return texelFetch(u_orig, clamp(p, ivec2(0), sz - 1), 0).rgb;
}
float diff3(vec3 a, vec3 b) { vec3 d = abs(a - b); return max(d.r, max(d.g, d.b)); }

// 境界（c = 後ろ側タイルの最初の列/行）での「不自然な段差」。step は境界を横切る向き、
// along は境界に沿う向き。ノイズ対策として境界に沿って前後 2 画素の平均をとる
float seamScore(ivec2 at, ivec2 step, ivec2 along)
{
    float s = 0.0;
    for (int k = -2; k <= 2; ++k) {
        ivec2 q  = at + along * k;
        vec3  a2 = origAt(q - step * 2), a1 = origAt(q - step), b0 = origAt(q), b1 = origAt(q + step);
        float across = diff3(a1, b0);
        float inner  = max(diff3(a2, a1), diff3(b0, b1));
        s += max(0.0, across - inner);
    }
    return s / 5.0;
}

void main()
{
    vec4  c = texture(tex, TexCoord);

    if (u_compareOn != 0) {
        vec2  size = vec2(textureSize(tex, 0));
        // 左端 / 上端からの距離（ピクセル）。GL の y は下から数えるので上下は反転する
        float d   = (u_compareDir == 0) ? gl_FragCoord.x : (size.y - gl_FragCoord.y);
        float cut = u_comparePos * ((u_compareDir == 0) ? size.x : size.y);
        if (abs(d - cut) <= u_lineW * 0.5) {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0);   // 境目の線
            return;
        }
        if (d < cut) c = texture(u_orig, vec2(TexCoord.x, 1.0 - TexCoord.y));
    }
    float y = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));

    if      (u_channel == 0) c.rgb = vec3(c.r);
    else if (u_channel == 1) c.rgb = vec3(c.g);
    else if (u_channel == 2) c.rgb = vec3(c.b);
    else if (u_channel == 3) c.rgb = vec3(y);

    if (u_zebraOn != 0) {
        // 右上がりの斜め縞（半周期ずつ塗る / 塗らない）
        bool stripe = mod(gl_FragCoord.x + gl_FragCoord.y, u_period) < u_period * 0.5;
        if (stripe && u_highOn != 0 && y >= u_high) {
            c.rgb = vec3(0.0);
        } else if (stripe && u_lowOn != 0 && y <= u_low) {
            c.rgb = vec3(0.15, 0.45, 1.0);
        }
    }
    if (u_seamOn != 0) {
        vec2  size = vec2(textureSize(tex, 0));
        ivec2 p    = ivec2(int(gl_FragCoord.x), int(size.y) - 1 - int(gl_FragCoord.y));   // 上から数えた座標
        float score = 0.0;
        // 縦の境界（列の切れ目）
        if (u_tileGrid.x > 1) {
            int cx = int(floor(float(p.x) / float(u_tileSize.x) + 0.5)) * u_tileSize.x;   // 最寄りの切れ目
            if (cx > 0 && cx < int(size.x) && abs(p.x - cx) < u_seamBand)
                score = max(score, seamScore(ivec2(cx, p.y), ivec2(1, 0), ivec2(0, 1)));
        }
        // 横の境界（行の切れ目）
        if (u_tileGrid.y > 1) {
            int cy = int(floor(float(p.y) / float(u_tileSize.y) + 0.5)) * u_tileSize.y;
            if (cy > 0 && cy < int(size.y) && abs(p.y - cy) < u_seamBand)
                score = max(score, seamScore(ivec2(p.x, cy), ivec2(0, 1), ivec2(1, 0)));
        }
        if (score > u_seamThr) {
            float t = clamp((score - u_seamThr) / max(u_seamThr, 0.01) * 0.5 + 0.5, 0.0, 1.0);
            c.rgb = mix(c.rgb, u_seamColor, t);
        }
    }

    FragColor = vec4(c.rgb, 1.0);
}
