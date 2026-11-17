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
  nvcc は同梱の `3rdparty/NVIDIAGPUComputingToolkit/CUDA/v13.0/bin` を優先、対象は sm_75 / 86 / 89 / 90 / 120。
  対象は `.pro` の `CUDA_SOURCES` に書いた 6 ファイル（decode / encode / histogram の _8bit・_10bit。8bit / 10bit は処理経路 = テクスチャ形式 RGBA8 / RGBA16 を表す）。
- 外部ライブラリは `3rdparty/` に同梱（.pro の `THIRDPARTY_DIR`）: CUDA Toolkit（`NVIDIAGPUComputingToolkit/CUDA/v13.0`。NVML の
  `nvml.lib` もここの lib/x64 を使う）、`ffmpeg`、`gpuimg`、`nvapi`。実行時の DLL（各 bin/）は PATH か exe と同じフォルダに置く
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
  - 表示の 10bit 化: `main.cpp` の既定フォーマットで色バッファ 10:10:10:2 を要求（アルファ 2bit も指定しないと WGL が 8bit 形式を選ぶ。実機確認済み）（VideoEngine・共有コンテキスト・ウィンドウ共通。
    共有には同じ形式が必要なので既定フォーマットで揃える）。取れなければ 8bit に落ちる。10bit 経路の RGBA16 をシーングラフがそのまま描くので
    10bit で出る。実際の深さは `sceneGraphInitialized` で読んで `PlayerController::setDisplayBits`（動画情報の Process/Render Depth の後ろ側）。`CUDA_VIEWER_DISPLAY_BITS=8` で従来に戻す。
    画面に出るまでの深さは `src/qmlui/displayprobe.*`（`-ldxgi -ld3d11`）が読んで PlayerController へ渡す:
    モニター出力 = `IDXGIOutput6::GetDesc1().BitsPerColor`（`setOutputBits`）、画面合成 = `DuplicateOutput1` の形式
    （SDR デスクトップは 8bit、HDR / 自動カラー管理が有効なら FP16。`setCompositionBits`）、全画面で合成を通るか =
    描き直しても Desktop Duplication に新しいフレームが来なければ直接出している（独立フリップ。`setComposed`）。
    起動時・`screenChanged`・`visibilityChanged`・アクティブになったときに読み直す。
    実機（RTX 5000 Ada）: ウィンドウ表示は SDR だと合成（Desktop Duplication で見える画面）は 8bit、自動カラー管理 / HDR を有効にすると FP16。全画面は直接出す。
    ただし NVIDIA のドライバ設定「Deep color for 3D applications」（`OGL_DEEP_COLOR_SCANOUT`、既定で有効）が効いていると、
    OpenGL の 10bit ウィンドウはドライバが合成の外で 10bit のまま出す（SDR・ウィンドウで ramp10_narrow.avi を 8bit 起動と見比べて確認済み）。
    これは公開 API で直接は取れないので、NVAPI（`3rdparty/nvapi`、NVIDIA 公開 SDK・MIT、`nvapi64.lib`）の DRS でこのアプリのプロファイル →
    ベースプロファイルの順に設定を読み（`nvDeepColorScanout`）、有効かつ色バッファ 10bit なら合成を数えない（`setDeepColorScanout`）色バッファは実行中に変えられないので 10bit 固定、
    8bit 素材は表示の値を中身どおり 8bit にする（`min(色バッファ, 素材 8/10)`）
  - 表示の 10bit 検証（検証用ビルドだけ）: qmake に `CONFIG+=display_verify` を付けると `src/qmlui/displayverify.*` を組み込み
    （`CUDA_VIEWER_DISPLAY_VERIFY`、`-ld3d11`）。Ctrl+Shift+D でカーソルの高さの 1 行（映像の表示枠）を
    ② 色バッファ（afterRendering で glReadPixels）と ③ 画面合成（Desktop Duplication `DuplicateOutput1` FP16）から読み、
    段数・平均幅・8bit 格子外の割合をログ `[verify]` に出す。テスト動画は RGB 10bit（FFV1 / gbrp10le、AVI → avidecode の GBRP10 経路）で
    4096×4096、上半分 1024 段（4px 幅）・下半分 256 段（16px 幅）:
    `ffmpeg -f lavfi -i "nullsrc=s=4096x4096:d=1:r=30,format=gbrp10le,geq=r='if(lt(Y,H/2),floor(X/4),floor(X/16)*4)':g='(同)':b='(同)'" -c:v ffv1 -level 3 ramp10.avi`
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
- 基本情報: ファイル名・デコーダ・デコード方式・解像度・**Bit Depth**（例 `10bit 4:2:0 · PQ`）・**Process/Render Depth**（例 `16bit / 10bit`、`16bit / 10bit → 8bit`。前 = 処理 `engine_->highBitDepth()` で 16 / 8、後 = 表示 = min(色バッファ `displayBits_`, 素材 10 / 8)、→ は画面に出るまでの一番浅い段（画面合成 `compBits_`〈合成を通るときだけ。全画面の直接出力・NVIDIA Deep color 有効なら数えない〉・モニター出力 `outputBits_`）がそれより浅いときだけ。値は `displayprobe.cpp`）・fps・総フレーム・現在フレーム・**Audio**（例 `AAC LC · 48 kHz · 6ch → 2ch`）。
  Bit Depth / Audio はファイルを開くとき `canUseGpuDecode` の中で一緒に読む（`srcPixel_` / `srcAudio_`）。
- 描画フレームレート: 実測 / 目標と直近 60 秒の推移（破線 = 目標、一時停止中は線が途切れる）。
  このアプリはコマを落とさず遅れるだけなので、取りこぼし数は出していない。
- GPU（プライマリが先頭）: GPU 全体 / 3D / CUDA / NVDEC / MEM の推移（直近 60 秒）。
  - 3D・CUDA は Windows の PDH（タスクマネージャーと同じ値）。取れない環境は隠す。
  - **HAGS 有効時は CUDA の負荷が 3D に入る**ので、CUDA のエンジンが無ければラベルを「3D+CUDA」にする。
  - GPU 全体 = 各エンジン（3D・CUDA・NVENC・NVDEC など）の最大。
- 推移の記録は QML 側だけで持つ。ファイルを閉じた・別ファイルを開いたらクリア。
- 入りきらないときは中をスクロール（収まっているときはホイールを映像のズームに渡す）。

### 性能 HUD（PerfHud.qml / PerfMeter、Ctrl+Shift+P・表示トグル「性能」）
- 映像の左下。段ごとの時間（直近 0.25 秒の平均・最大 ms）とフレーム間隔の推移（直近 300 フレーム、破線 = 持ち時間）、NVDEC 使用率。
- 計測は `src/qmlui/perfmeter.*`（`VideoEngine::perf()`）。OFF の間は各フックが即 return（GL クエリも CUDA イベントも出さない）。
  デコード = `decode_thread::decode_time_ms`（get_decode_image の実時間）、変換 = interop_stream の CUDA イベント、
  画像処理 / 表示ツール = GL のタイムスタンプ（`glQueryCounter`）、ヒストグラム = CPU 実時間（同期するため）、
  描画スレッド = `upload` 全体の CPU 時間、フレーム間隔 = `upload` の呼ばれた間隔。GPU の結果は 4 フレームのリングで待たずに回収。
- `PlayerController::perf` は 0.25 秒ごと（`perfTimer_`）。持ち時間 = 1000 / (fps × 再生速度)。ON/OFF は ini `Info/perf_hud`。
- デコードの時間は「待ち」（NVDEC は先読みで並行）なので限界は分からない。NVDEC 使用率から上限 fps の目安（今の fps ÷ 使用率）を出す。
  フレーム間隔の色は持ち時間の 5% 超えから（再生は fps に合わせて進むので、間隔 ≒ 持ち時間が正常）。
- 実測（RTX 4080 SUPER、16K 8bit 4:2:0 30fps）: GPU の段の合計 約 2.6ms、描画スレッド 約 4.9ms、NVDEC 80%（上限 約 37fps）→ 16K の高 fps はデコードが限界
- inline component（Stage）は外側の id を参照しないよう、持ち時間を `budget` で渡している。

### プロパティ（ファイル → プロパティ…、Ctrl+I）
- MediaInfo 風のセクション表示（全般 / 映像 #n / 音声 #n / 字幕・その他）。値は選択してコピー可。
- 「クリップボードにコピー」「名前を付けて保存…」（`<元の名前>_info.txt`）。

## 5.5 マルチ GPU デコード（DecodePanel / DecodeSettings、ファイル → デコード設定…）
- GPU ごとの重み `GPUInfo::decode_weight`（0..10、-1 = 未設定 = 表示用 GPU だけ 10。ini の GPU グループ `decode_weight`）。
  割り振りは `src/videoprocess/decode_gpu_map.h` の `assignDecodeTiles`（最大剰余法、表示用 GPU のタイルが先頭、
  そのコーデックを NVDEC で扱えない GPU は外す: AV1 は sm_86 以降）。結果は `VideoInfo.tile_gpu`、動画情報の Decode Mode に `Decode GPU: 0×3 + 1×1`。
- `nvgpudecode`: 別 GPU のタイルはその GPU の FFmpeg デバイスコンテキスト（`primary_ctx`、`remote_hw_ctx`）で cuvid を開く。
  `publish_decode_frame` → `transfer_remote` で「その GPU → メインメモリ（ページ固定、`cudaHostAllocPortable`、`kStage`=3 枠）→
  表示用 GPU のリング（`VideoDecorder::r_y / r_uv`、hw_frames と同じ段数）」の 2 段で送る。
  D2H はその GPU の既定ストリーム（cuvid の出力コピーの後）、H2D と `decode_ready` は表示用 GPU の非ブロッキング `xfer_stream_`
  （表示用 GPU のタイルの完了 `local_ev_` と D2H の完了を待ってから）。同じ中継枠を次に使う前に前回の H2D 完了をその GPU に待たせる。
  GUI 側は従来どおり `decode_ready` だけ待つ。
  最初は `cudaMemcpy3DPeerAsync`（P2P 無しはドライバがメインメモリ経由）だったが、RTX 4080 SUPER + RTX 2070 で約 2.4GB/s で頭打ち
  （16K 4 タイル: 3:1 で 48fps、2:2 で 23fps）だったので 2 段に変えた。
- 設定はファイルを開くときに読む。「OK して開き直す」は `PlayerController::reopenForDecodeSettings`（開き直して `reopen_frame_` へ高精度シーク）。
- 性能 HUD: `perf.nvdecs`（デコードに使っている GPU ごとの NVDEC）、上限の目安は最大の使用率から。
  「転送」= D2H の時間（その GPU の CUDA イベント、`decode_thread::transfer_time_ms`）と速度 GB/s。マルチ GPU デコードのときだけ表示。
- 実測の前提（RTX 4080 SUPER 単体、16K 8bit 30fps）: NVDEC 80%、2 倍速で 37fps 頭打ち → 60fps にはデコードの分担が必要。サブは RTX 2070（Turing: AV1 不可）。

## 6. エンコード設定（EncodePanel / EncodeController）
- 開いている間は後ろの画面を 1 枚のテクスチャに焼いて固定（`ShaderEffectSource live:false, hideSource`）。
- 録画フォーマット → コーデックの順（コンテナで選べるコーデックが変わる）。
- エンコード中は進捗と速度（fps）を表示。閉じると、設定を開いた時点のフレームと再生状態へ戻る。
- マルチ GPU 表: 重み・タイル数と、GPU ごとの使用率の推移（2×2: GPU 全体 | NVENC / 3D+CUDA | VRAM、直近 60 秒、開くたびにクリア）。
  NVENC に余裕があり GPU 全体が高いなら、エンコーダより手前（デコード・処理・変換）が詰まっている。

## 7. 操作・状態

- ショートカット: Space（再生 / 一時停止）、F11（映像だけ全画面）、Esc（全画面解除）、Ctrl+O（開く）、Ctrl+I（プロパティ）、Ctrl+Shift+P（性能 HUD）。
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

## 9. 10bit 対応の実機検証（RTX 5000 Ada、2026-11）

テスト動画（RGB 10bit、FFV1 / gbrp10le の AVI。CPU デコードの GBRP10 経路で RGB の値がそのまま入る＝リミテッドレンジ変換なし）:
- `ramp10.avi`: 4096×4096、上半分 0..1023 の 1024 段（4px 幅）、下半分 256 段（16px 幅）。検証コマンドの段数を数える用
- `ramp10_narrow.avi`: 4096×2160、256..358 の範囲で上半分 103 段（40px 幅）、下半分 26 段（160px 幅）。目で見比べる用
- 作り方は §3 の「表示の 10bit 検証」の ffmpeg コマンド（narrow は式を `256+floor(X/40)` / `256+floor(X/160)*4` に）

結果:
| 条件 | ② 色バッファ | ③ 画面合成 | 見た目（narrow） | Process/Render Depth |
|---|---|---|---|---|
| 8bit 起動（`CUDA_VIEWER_DISPLAY_BITS=8`） | 256 段・8bit 外 0% | 256 | 上下同じ縞 | `16bit / 8bit` |
| 10bit・全画面 | 1024 段・8bit 外 42%（3840px に縮小表示） | 読めない（直接出力でフレームが来ない） | 上が滑らか | `16bit / 10bit` |
| 10bit・ウィンドウ・SDR・出力 10bpc | 10bit | 256（8bit 形式） | 上が滑らか（Deep color） | `16bit / 10bit` |
| 10bit・ウィンドウ・自動カラー管理 or HDR | 980 段 | 980 段（FP16） | 上が滑らか | `16bit / 10bit` |
| 出力 8bpc（Deep color 有効） | （色バッファ 8bit になる） | — | 上下同じ縞 | `16bit / 8bit` |

分かったこと:
- 色バッファは `setAlphaBufferSize(2)` まで指定しないと WGL が 8bit の形式を選ぶ（10:10:10:2）。
- SDR のデスクトップ合成は 8bit。ウィンドウ表示で 10bit が出るのは NVIDIA の Deep color（ドライバが合成の外で出す。Desktop Duplication からは見えない）か、
  自動カラー管理 / HDR（合成が FP16）のとき。全画面は合成を通らない（独立フリップ）。
- 出力 8bpc にすると、ドライバは 10bit の色バッファを渡さない（アプリの出力自体が 8bit）。
- スポイトの値は RGB（リミテッドレンジの YUV を 0..1023 に広げた値）なので、YUV の 1 段が RGB では約 1.17 段になり、6〜7 段に 1 回値が飛ぶ（正常）。
- Deep color のオフ（NVIDIA Profile Inspector で `0x2097C2F6` を Disabled）は未確認。通常触らない隠し設定（既定で有効）なので省略した。
  オフのときの表示は NVAPI 追加前と同じ判定（SDR・ウィンドウで `→ 8bit`）になる。

## 10. 今後の候補（未着手）

- トーンカーブの背景にヒストグラム、カーソル位置の値表示
- スポイトのピン留め・記録
- ズーム表示を「1 画素 = ○px」に
- ベンチマークモード（性能 HUD は実装済み。一定区間の平均・最悪値・パーセンタイルを CSV に記録する機能は未）
- マルチ GPU デコードの続き（実装済みは 5.5。残り: 10bit の詰め直しで転送量を 10/16 に、nvcomp は効果を測ってから）
- ディザ付きの画面表示（表示は 10bit 化済み。8bit の色バッファしか取れない環境向け。8bit 化して保存・書き出すときのディザは対応済み。色の行列は BT.709 に統一済み。HDR の PQ / HLG 引き継ぎは非対応の方針）
- 2 ファイル比較
- Widgets 版との機能差の解消
