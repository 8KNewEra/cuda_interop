# QML 版 UI 引き継ぎメモ

CUDA + OpenGL + FFmpeg による GPU 専用の 8K / 16K タイル動画ビューア・エンコーダ（Qt 6.9）。
このメモは Qt Quick 版 UI（`--qml`）で実装した内容の全体像と、触るときの注意点をまとめたもの。
細部は各ファイル先頭のコメントを参照（コメントは現状に合わせて更新済み）。

---

## 1. 起動とビルド

- `--qml` 引数、または環境変数 `CUDA_VIEWER_UI=qml` で Qt Quick 版。なしなら従来の Widgets 版（`MainWindow` / `GLWidget`）。
- ファイルを追加・削除したら **qmake を実行してから再ビルド**（`cuda_interop.pro`・`qml.qrc`・`qml/qmldir` の 3 か所に登録が要る）。
- Windows 専用のリンク: `opengl32`、`nvml`、`pdh`（GPU の 3D / CUDA 使用率）。
- CUDA カーネル（`cuda_kernels/*.cu`）は qmake の extra compiler で nvcc に通す（以前の `*.bat` + `obj_out/*.obj` は不要）。
  nvcc は同梱の `NVIDIAGPUComputingToolkit/CUDA/v13.0/bin` を優先、対象は sm_75 / 86 / 89 / 90 / 120。
  対象は `.pro` の `CUDA_SOURCES` に書いた 3 ファイル（decode_8bit / encode_8bit / histogram_8bit。8bit / 10bit は処理経路 = テクスチャ形式 RGBA8 / RGBA16 を表す）。
  起動口は `src/imageprocess/cuda_imageprocess.cpp`（`cudaLaunchKernel`）。出力はビルドフォルダの `cuda_obj/debug|release/`。
- 対象 GPU は NVIDIA のみ（CUDA 必須）。UI 上の GPU 名からは先頭の "NVIDIA " を省いている。

## 2. 全体構成

```
main.cpp ── PlayerController（context property "player"）
              ├─ VideoEngine      … 描画・処理部（自前 GL コンテキスト、CUDA interop、シェーダ、NVENC 受け渡し）
              ├─ EncodeController  … エンコード設定（"encoder"）
              ├─ ScreenshotSettings… 画像保存設定（"shotSettings"）
              ├─ FilterKernel ×2   … ガウシアン / 平均化の係数（player.gaussianKernel / averagingKernel）
              ├─ MediaProbe        … プロパティ画面の情報（player.mediaInfo）
              └─ decode / fps / info スレッド（Widgets 版 MainWindow と同じ手順）
VideoItem / HistogramItem … VideoEngine のテクスチャをコピーせずシーングラフへ載せる
WindowChrome              … OS のタイトルバーを消して QML のヘッダーをタイトルバーにする（Windows）
```

- 画像処理・表示ツールの設定はすべて `g_AppSettings`（`src/main/__global__.h`）に直接持つ。
  変更したら `engine_->filter_change_flag = true` → 次の描画で uniform を送り直す。
- 終了時に `logfile_control` のデストラクタが `setting.ini` へ保存（画像処理は `[Filter]` セクション）。

### C++（`src/qmlui/`）

| ファイル | 役割 |
|---|---|
| playercontroller | QML の窓口。再生操作・範囲・ジャンプ・フィルタ設定・情報表示・エンコードの段取り |
| videoengine | 1 フレームの処理パイプライン（下記 3 章）。スポイト・ヒストグラム・画像保存・エンコード入力 |
| videoitem | VideoItem（映像表示・ズーム / パン）、HistogramItem |
| encodecontroller | エンコード設定の値・制約・GPU 割り当て。`normalize()` で制約の連鎖を一括解決 |
| screenshotsettings | 画像保存設定（PNG / TIFF / JPEG など） |
| filterkernel | 畳み込み係数（サイズ 3/5/7/9、σ、正規化、直接編集） |
| tonecurve | トーンカーブの点 ⇔ 文字列 ⇔ 256 段 LUT（単調 3 次補間、Fritsch–Carlson） |
| mediaprobe | FFmpeg でファイルの詳細情報を読む（別スレッド） |
| windowchrome | フレームレスでもスナップ・影・角丸を残すための WM_NCCALCSIZE / WM_NCHITTEST 処理 |

### QML（`qml/`）

| ファイル | 役割 |
|---|---|
| Main.qml | ルート。ヘッダー（メニュー）/ 映像 / 操作パネル、映像上の重ね表示、ダイアログ |
| ControlPanel / RangeBar | 下部の操作パネル、再生位置＋範囲スライダー |
| InfoPanel | 動画情報（左上）。基本情報 → 描画 fps → GPU ごとの使用率の推移 |
| HistogramPanel / ZoomNavigator | ヒストグラム、ズーム位置 |
| FilterPanel | 右ドロワー。タブ「画像処理」「表示ツール」。各 *Params.qml が設定欄 |
| EncodePanel / ScreenshotPanel / PropertiesPanel | 画面全体に重ねるダイアログ |
| KernelEditor / KernelParams | 係数の表示と直接編集 |
| ToneCurveParams | トーンカーブ編集（C++ の LUT をそのまま描く） |
| Sparkline | 推移グラフ（動画情報・エンコード設定で共用） |
| MiniSlider / NumberSpin / ValueRow / Segmented / Select / ColorSwatches / ToggleSwitch / Icon / IconButton / ChipButton / AppMenu | 共通部品 |
| Theme | 色・フォント定数（シングルトン）。`seamColors` は C++ の `kSeamColors` と同じ順 |

## 3. 描画パイプライン（VideoEngine::FBO_Rendering）

```
入力テクスチャ（CUDA が書く。上下逆）
  → 処理パス（ON のものだけ。fbo / tempfbo を交互に使い、最後が fbo に来るよう開始側を決める）
       colorcorrect.frag … 画像補正・色補正・トーンカーブを 1 パス（いちばん先）
       gaussian.frag     … ぼかし / シャープ化（アンシャープマスク）
       sobel.frag        … エッジ / ピーキング
       averaging.frag    … 平均化
       （全部 OFF なら素通し 1 パス）
  → fbo = 処理結果。ヒストグラム・画像保存・エンコードはここを読む
  → 表示専用パス zebra.frag（比較 → チャンネル → ゼブラ → タイル境界）: fbo → tempfbo
  → displayTexture()（表示ツールがあれば tempTexture、なければ fbo）を VideoItem が表示
```

- **上下反転は 1 パス目だけ**（`vao`）。2 パス目以降は `vaoNoFlip` と `texelSize.y` の符号反転。
  全パスで反転するとパス数が偶数のとき逆さまになる（Widgets 版 glwidget も同じ修正済み）。
- `zebraProg_` / `Zebra_program` は名前に反して**表示専用パス全体**を担当。
- **処理経路は 2 本**（`VideoEngine::rgba16_`、素材ごとに `upload` が自動で切り替え。テクスチャを作り直す）
  - 8bit 経路: 入力・fbo・tempfbo が RGBA8。カーネルは `cuda_kernels/*_8bit.cu`（Widgets 版は常にこちら）
  - 10bit 経路: 9bit 以上の素材（NVDEC の P010、CPU デコードの 10bit プレーン）。テクスチャは RGBA16、
    カーネルは `cuda_kernels/*_10bit.cu`（引数の構造体は `tiles_10bit.h`）。シェーダは共通（float で読み書き）
  - 画像保存: 10bit 経路の PNG / TIFF は、設定 `screenshot_depth`（画像保存設定の「ビット深度」。表記は 10bit / 8bit、8bit の動画は 8bit のみ。既定 "16"）が 16 なら
    fbo の RGBA16 をそのまま 16bit で保存（`gpuimg::Image::rgba16`）。
    JPEG / BMP は形式上 8bit なので RGBA8 に四捨五入（`rgba16_to_rgba8_tex`）。保存用バッファは経路に合わせて 4 / 8 バイト/画素
  - 表示の 10bit 化: `main.cpp` の既定フォーマットで色バッファ 10:10:10 を要求（VideoEngine・共有コンテキスト・ウィンドウ共通。
    共有には同じ形式が必要なので既定フォーマットで揃える）。取れなければ 8bit に落ちる。10bit 経路の RGBA16 をシーングラフがそのまま描くので
    10bit で出る。実際の深さは `sceneGraphInitialized` で読んで `PlayerController::setDisplayBits`（動画情報の Process/Render Depth の後ろ側）。`CUDA_VIEWER_DISPLAY_BITS=8` で従来に戻す
  - 8bit 化のディザ: 10bit 経路から 8bit に落とす 2 本（`rgba16_to_nv12_split_tex` / `rgba16_to_rgba8_tex`）は引数 dither で
    ±0.5 段階のノイズ（座標とチャンネルのハッシュ。フレーム間で同じ）を足してから四捨五入。設定は画像保存 `screenshot_dither`（"on"/"off"、
    ini `Screenshot/screenshot_dither`）と書き出し `EncodeSettings::dither`（ini `Encodesetting/dither`）。既定はどちらもあり。
    設定画面の「ディザ」行は 10bit の動画を 8bit で出すときだけ表示（`showDither`）
  - 書き出し: `save_encode(h, w, srcRgba16)`。`encode_out_10bit()`（10bit 経路 × HEVC / AV1 × 設定 `output_depth`=10）なら
    P010（`rgba16_to_p010_split_tex`、BT.709、色差 2×2 平均、HEVC は Main10、VUI に BT.709 / limited）、
    それ以外は NV12 8bit（`rgba16_to_nv12_split_tex`）。NV12 も P010 も色差は 2×2 平均・四捨五入（8bit 経路の `rgba_to_nv12x*` も同じ）。設定画面の「プロファイル」（HEVC: Main / Main 10、AV1: Main（8bit / 10bit）、H.264: 8bit のみ）と ini `Encodesetting/output_depth`。ヒストグラムは 1024 段（`histogram_10bit.cu` / `HistData10`、統計は 0..1023、`histStats().bins`。16bit→10bit はスポイトと同じ四捨五入）、スポイトは 0..1023 を返す（`bits: 10`）
  - 色の行列は全経路 BT.709 に統一（デコード 8bit / 10bit、エンコード NV12 / P010）。レンジは limited（CPU デコードの yuv422p 8bit だけ旧実装どおり full）。
    書き出しは 8bit / 10bit とも VUI に BT.709（行列・原色・伝達特性）/ limited を書く。HDR（PQ / HLG）は非対応の方針で、SDR BT.709 として扱う
- コンテキストが 2 つ（engine と Qt Quick）あるので GPU 上の順序はフェンス（`frameFence` / `displayDoneFence`）で渡す。
  render loop は basic（GUI スレッド）固定。
- トーンカーブの LUT は 1D テクスチャ（ユニット 2、256×RGB、RGB 全体と各チャンネルを合成済み）。
- ソーベルの `u_mode`: 0 = Widgets 版の従来動作、1 = エッジのみ、2 = ピーキング。QML 版は `sobel_mode`（0/1）を 1/2 にして渡す。

## 4. 画像処理パネルの機能

### 画像処理タブ（映像そのものを変える。ヒストグラム・画像保存・エンコードに入る）
- **画像補正**: 明るさ・コントラスト・シャドウ・ハイライト・ガンマ（`colorEnabled` が ON/OFF。名前は ini 互換のため）
- **色補正**: 色温度・ティント・彩度、R/G/B ごとのゲイン・オフセット・ガンマ（タブで切り替え）
- **トーンカーブ**: RGB 全体 / R / G / B。点は最大 16、両端は上下のみ。ini に "入力:出力,..." で保存
- **ガウシアン**: サイズ・σ・正規化・係数の直接編集、モード（ぼかし / シャープ化と強さ）
- **ソーベル**: エッジのみ / ピーキング（しきい値・色）
- **平均化**: サイズ・正規化・係数の直接編集

### 表示ツールタブ（画面表示だけ。ヒストグラム・画像保存・エンコードには入らない）
- **比較表示**: 補正前 / 補正後を左右または上下で分割。境目は画面上でドラッグ（0.1% 単位）
- **チャンネル表示**: R / G / B / 輝度を白黒で
- **タイル境界チェック**: 境界線とタイル番号、つなぎ目の段差の強調（色を選べる）。タイル分割の動画のみ
- **スポイト**: マウス位置の画素値（補正後 / 補正前 / 両方）。描画フレームごとに CUDA の非同期コピーで取得
- **ゼブラ**: 白飛び（黒の縞）/ 黒つぶれ（青の縞）

### 共通仕様
- 大項目が OFF の間は小項目を触れない。スライダーのダブルクリックで既定値へ戻る。
- パラメータは `player.filterParams`（読み）/ `player.setFilterParam(key, v)`（書き）。
  キー: sobelMode, peakingThreshold, peakingColor, zebraHighOn, zebraHighLevel, zebraLowOn, zebraLowLevel,
  gaussianMode, sharpenAmount, channelMode, compareDir, comparePos(0..1000), seamGrid, seamHighlight,
  seamThreshold, seamGridColor, seamHighlightColor, eyedropperMode, brightness, contrast, shadows,
  highlights, gamma100, saturation, temperature, tint, gainR/G/B, offsetR/G/B, gammaR/G/B
- 係数の直接編集は保存しない（サイズ・σ・正規化だけ ini に残る）。

## 5. 情報表示

### 動画情報（InfoPanel）
- 基本情報: ファイル名・デコーダ・デコード方式・解像度・**Bit Depth**（例 `10bit 4:2:0 · PQ`）・**Process/Render Depth**（例 `16bit / 10bit`。前 = 処理 `engine_->highBitDepth()` で 16 / 8、後 = 表示 = ウィンドウの色バッファ `displayBits_`）・fps・総フレーム・現在フレーム・**Audio**（例 `AAC LC · 48 kHz · 6ch → 2ch`）。
  Bit Depth / Audio はファイルを開くとき `canUseGpuDecode` の中で一緒に読む（`srcPixel_` / `srcAudio_`）。
- 描画フレームレート: 実測 / 目標と直近 60 秒の推移（破線 = 目標、一時停止中は線が途切れる）。
  このアプリはコマを落とさず遅れるだけなので、取りこぼし数は出していない。
- GPU（プライマリが先頭）: GPU 全体 / 3D / CUDA / NVDEC / MEM の推移（直近 60 秒）。
  - 3D・CUDA は Windows の PDH（タスクマネージャーと同じ値）。取れない環境は隠す。
  - **HAGS 有効時は CUDA の負荷が 3D に入る**ので、CUDA のエンジンが無ければラベルを「3D+CUDA」にする。
  - GPU 全体 = 各エンジン（3D・CUDA・NVENC・NVDEC など）の最大。
- 推移の記録は QML 側だけで持つ。ファイルを閉じた・別ファイルを開いたらクリア。
- 入りきらないときは中をスクロール（収まっているときはホイールを映像のズームに渡す）。

### プロパティ（ファイル → プロパティ…、Ctrl+I）
- MediaInfo 風のセクション表示（全般 / 映像 #n / 音声 #n / 字幕・その他）。値は選択してコピー可。
- 「クリップボードにコピー」「名前を付けて保存…」（`<元の名前>_info.txt`）。

## 6. エンコード設定（EncodePanel / EncodeController）
- 開いている間は後ろの画面を 1 枚のテクスチャに焼いて固定（`ShaderEffectSource live:false, hideSource`）。
- 録画フォーマット → コーデックの順（コンテナで選べるコーデックが変わる）。
- エンコード中は進捗と速度（fps）を表示。閉じると、設定を開いた時点のフレームと再生状態へ戻る。
- マルチ GPU 表: 重み・タイル数と、GPU ごとの使用率の推移（2×2: GPU 全体 | NVENC / 3D+CUDA | VRAM、直近 60 秒、開くたびにクリア）。
  NVENC に余裕があり GPU 全体が高いなら、エンコーダより手前（デコード・処理・変換）が詰まっている。

## 7. 操作・状態

- ショートカット: Space（再生 / 一時停止）、F11（映像だけ全画面）、Esc（全画面解除）、Ctrl+O（開く）、Ctrl+I（プロパティ）。
- `win.busy` = ファイルを開いていて `controlsEnabled` が false（高精度シーク・画像保存などの重い処理中）。
  busy 中に使えるのは「ファイルを閉じる」とウィンドウ操作だけ。復帰しない場合も「ファイルを閉じる」で抜けられる。
- `win.modalOpen`（エンコード設定・画像保存設定・プロパティ）中はヘッダーとショートカットを止める。
- ズームは最大 1 画素 = 32px。拡大時は画素グリッドを描き、スポイトの読み取り画素を枠で囲む。

## 8. 実装上の注意（ハマりどころ）

- **inline component はファイル直下に置き、外側の id を参照しない**（Qt 6.9 で解決できないことがある）。
  外側を参照したい部品は Repeater か別ファイルにする。
- **Repeater に毎回新しくなる配列を直接渡さない**。`player.gpus` / `encoder.gpus` は更新のたびに新しい配列なので、
  model は台数（`length`）にして中身は `list[index]` から読む（行が作り直されてグラフがちらつくのを防ぐ）。
- 推移グラフは配列を作り直して代入する（中身を書き換えても変更通知が出ない）。
- `Item.scale` など既存プロパティ名と衝突する名前を付けない（NumberSpin は `factor`）。
- StyledText は空白が縮むので桁そろえは `&nbsp;`。
- 色は文字列＋opacity で扱い、`Qt.color()` は使っていない。
- `windows.h` を読む場所では `NOMINMAX`、`std::max` は `(std::max)` で書く（info_thread.cpp）。
- ファイルは CRLF。
- 動作確認の代わりに手元で QML をモック（PySide6）で描画して確認してきた。新しめの Qt で入ったプロパティ
  （Icon.qml の `preferredRendererType` など）は古い Qt では読めないので注意。

## 9. 今後の候補（未着手）

- トーンカーブの背景にヒストグラム、カーソル位置の値表示
- スポイトのピン留め・記録
- ズーム表示を「1 画素 = ○px」に
- 性能 HUD / ベンチマークモード（「NVIDIA 限定・最速」の打ち出し用）
- ディザ付きの画面表示（表示は 10bit 化済み。8bit の色バッファしか取れない環境向け。8bit 化して保存・書き出すときのディザは対応済み。色の行列は BT.709 に統一済み。HDR の PQ / HLG 引き継ぎは非対応の方針）
- 2 ファイル比較
- Widgets 版との機能差の解消
