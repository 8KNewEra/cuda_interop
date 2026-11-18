# マルチ GPU デコード

タイル分割された動画（2 / 4 / 8 タイル）のタイルを複数の GPU の NVDEC でデコードし、表示用 GPU に集めて表示する機能。
QML 版 UI の引き継ぎメモ（`HANDOFF_QML_UI.md`）の 5.5 の詳細。

## 1. 使い方

- ファイル → デコード設定…（`qml/DecodePanel.qml`）で GPU ごとの重み（0..10）を決める。重み 0 の GPU は使わない。
  タイルの動画を開いているときは「このファイル」に割り振りの見込みが出る。反映は次に開いたとき（「OK して開き直す」で今の位置から開き直す）。
- 「転送の圧縮」は別 GPU から表示用 GPU へ送るタイルを nvCOMP で圧縮する（可逆）。推奨は ANS・差分あり（既定）。
  使う GPU が 1 枚のときは転送が無いので「なし」表示で触れない（保存値はそのまま）。
- **タイル数の目安 = GPU の数 × NVDEC の基数**。1 タイル（= 1 デコードセッション）は NVDEC を 1 基しか使わない。
  例: RTX 5000 Ada（NVDEC 2 基）×4 なら 8 タイル。4 タイルだと各 GPU の NVDEC が半分遊ぶ（性能 HUD に「NVDEC 1/2 基のみ使用」）。

## 2. ファイル

| ファイル | 役割 |
|---|---|
| `src/videoprocess/decode_gpu_map.h` | タイルの割り振り（`assignDecodeTiles`）、重み、コーデック対応 |
| `src/videoprocess/nvgpudecode.*` | デコーダ作成（別 GPU のコンテキスト）、転送（`transfer_remote` / `transfer_compressed`）、NVDEC の基数、デコード待ちの内訳 |
| `src/videoprocess/xfer_codec.*` | 圧縮転送（`XferCodec`。1 つ = 1 枚の別 GPU） |
| `cuda_kernels/xfer_compress.cu` | 圧縮転送のカーネル（前処理・詰め・取り込み・差分戻し） |
| `src/videoprocess/decode_thread.*` | `VideoDecorder` の転送用フィールド、HUD 用シグナル、デコードの先行 |
| `src/qmlui/decodesettings.*` / `qml/DecodePanel.qml` | 設定画面 |
| `src/qmlui/perfmeter.*` / `qml/PerfHud.qml` | 性能 HUD |
| `3rdparty/nvcomp/` | NVIDIA nvCOMP 6.0（ヘッダ・lib・DLL。DLL はビルド後に exe の横へコピー） |

## 3. タイルの割り振り（decode_gpu_map.h）

- 重み `GPUInfo::decode_weight`（ini の GPU グループ `decode_weight`）。未設定（-1）は表示用 GPU だけ 10、ほか 0 = 従来どおり 1 枚。
- そのコーデックを NVDEC で扱えない GPU は外す（AV1 は sm_86 以降）。使える GPU が無ければ全部を表示用 GPU。
- 最大剰余法。端数は 余りの大きい順 → 重みの大きい順 → 表示用 GPU → 番号の小さい順。
- 戻り値はタイルごとのデバイス番号で、表示用 GPU のタイルが先頭。結果は `VideoInfo.tile_gpu`。
  動画情報の「Decode GPU」に `3×2 + 0×2 + …`（GPU 番号×タイル数）。

## 4. 1 コマの流れ

```
decode_thread::processFrame
  └ nvgpudecode::get_multidecode_image
      ├ read_into_pending / pump_stream（タイルごと。cuvid は avcodec_send_packet の中で
      │   パース → NVDEC 投入 → 完了待ち → 出力コピー発行 まで進む）
      └ publish_decode_frame
          ├ 表示用 GPU のタイル: hw_frames の面をそのまま Frame.d_y / d_uv に
          ├ 別 GPU のタイル   : r_y / r_uv（表示用 GPU 上のリング）を Frame に載せ、transfer_remote で送る
          └ decode_ready（全タイルの完成）を記録 → GUI は upload でこれだけ待つ
```

- 別 GPU の cuvid は、その GPU の FFmpeg デバイスコンテキスト（`primary_ctx=1`、`remote_hw_ctx`）で開く。
  Runtime API と同じ primary context なので、Runtime のイベントで順序を取れる。
- 転送の準備（`alloc_remote_tile`）はそのタイルの最初のコマで行う。転送先リングは `hw_frames` と同じ段数（`ringSize` = 6）。

## 5. 転送

### 5.1 圧縮しない（2 段コピー）

```
別 GPU  既定ストリーム: [cuvid 出力コピー] → (h2d_ev_[s] 待ち) → D2H → h_y/h_uv[s]（ページ固定）
表示用 GPU xfer_stream_: (local_ev_[s] と D2H 完了を待つ) → H2D → r_y/r_uv[ringNo] → h2d_ev_[s] → decode_ready
```

- 中継はメインメモリの `kStage` = 3 枠を順に使う。同じ枠の再利用は前回の H2D 完了（`h2d_ev_`）を待つ。
- `cudaMemcpy3DPeerAsync`（P2P 無しの環境ではドライバがメインメモリ経由）は約 2.4GB/s で頭打ちだったので 2 段にした。
- 帯域は PCIe で決まる（RTX 2070 on Gen3 x4 で約 3GB/s、Gen4 x16 で約 26GB/s）。

### 5.2 圧縮する（XferCodec）

```
別 GPU（組 k のストリーム。2 組を交互）
  (既定ストリームの後) → xfer_fwd → nvCOMP 圧縮 → [cmpEnd] → (stageFree 待ち) → [sendBeg]
  → xfer_pack（メインメモリ hData_[s] へカーネルが直接書く）＋ ヘッダを DMA → [encEnd]
表示用 GPU xfer_stream_
  (local_ev_ と encEnd を待つ) → xfer_fetch（ヘッダと見込み分は DMA、超過分はカーネル）
  → nvCOMP 展開 → xfer_inv（差分を戻して r_y/r_uv[ringNo]）→ h2d_ev_[s] → decode_ready
```

- **詰めた並び**: タイルごとに Y（w×h）、UV（w×h/2）を 16 バイト境界で並べ、64KB のチャンクに切って一括圧縮。
- **前処理（差分あり）**: 行の中で Y は 1 つ左、UV は 2 つ左（U/V 交互）との差を bits ビットで巡回させ、ジグザグで正の数に。
  滑らかな絵ほど 0 付近に集まって縮む。可逆。P010 は >>6 して 10bit で扱う（12bit 素材は下位 2bit が落ちるが、表示も 10bit なので実害なし）。
- **xfer_pack**: チャンクの位置を 1 スレッドで決め（`hdr[0]` = 数、`[1]` = 合計、`[2..2+n)` = 大きさ、`[2+n..)` = 位置）、
  ページ固定・マップ済みのメインメモリへカーネルが直接書く。CPU は圧縮後の大きさを知らずに済み、実際のバイト数だけ PCIe を流れる。
  ブロック数は 32 に絞り、次のフレームの圧縮と並べる。
- **xfer_fetch**: 見込み（直近の実績 × 1.25 + 1MB、最初は全体の 1/4）までは `cudaMemcpyAsync`（コピーエンジン。SM を使わない）、
  超えた分だけカーネルがメインメモリから読む。見込みが外れても速さだけの問題。
- **2 組（kSets）**: 別 GPU の作業領域とストリームを 2 組持ち、前のフレームの送出と次の圧縮を重ねる。
  既定ストリームに積まないので cuvid の出力コピーも待たせない。
- 初期化に失敗したらそのファイルの間は圧縮なし（5.1）に戻る。
- Bitcomp だけはネイティブ API で全体を 1 本で圧縮する。圧縮後の大きさを `bitcompGetCompressedSizeAsync` で GPU 上に受け、
  `xfer_split` で 64KB の仮想チャンクに切り分けてから同じ `xfer_pack` / `xfer_fetch` を使う（仮想チャンクは隙間なく並ぶので、
  表示用 GPU では取り込んだ先頭からそのまま展開できる）。

### 5.3 方式

| 方式 | 結果（RTX 5000 Ada ×4、16K 8bit） |
|---|---|
| ANS ＋ 差分 | 推奨。圧縮率 約 9、展開 約 2.6ms |
| ANS・差分なし | 圧縮率が落ちて取り込みが重い（展開 約 6.4ms） |
| Bitcomp | バッチ API では圧縮の呼び出しが CPU を約 4ms 止めた（「転送の発行」に出る）。ネイティブ API（`nvcomp/native/bitcomp.h`。プランを組ごとに作ってストリームに結び付け、全体を 1 本で圧縮）に切り替えた。効果は要計測 |
| Cascaded | 平坦な絵では縮むが、一般には ANS より低い |

LZ4 / Snappy は画像の差分と相性が悪く、GDeflate / Zstd は GPU での圧縮が遅いので入れていない。
圧縮率を上げるなら方式より前処理（左・上・左上からの予測）が効く見込み。

## 6. 同期とバッファの関係（壊さないための条件）

- `hw_frames` / `r_y` / `r_uv` のリング（6 段）: 同時に使うのは GUI の未受け取り ≦ 2 ＋ GUI の GPU 処理中 2（`VideoEngine::kMaxFramesInFlight`）
  ＋ デコード中 1 = 5。先行の上限を上げるならリングも増やす。超えるとログに `RING OVERRUN`（防ぐ仕組みは無い）。
- メインメモリの枠（`kStage` = 3）: 再利用は前回の取り込み完了（`h2d_ev_[s]`）を待つ（圧縮なしは別 GPU の既定ストリーム、
  圧縮ありは組のストリームで待つ）。
- 圧縮ありの別 GPU 側: 組 k を使う前に、既定ストリームに組 k の前回の完了（`k.done`）を待たせる。
  以降の cuvid の出力コピーが、まだ読んでいる面を上書きしないための保険（面はリングが一周するまで残るので、実際にはまず待たない）。
- `decode_ready` は常に `xfer_stream_` の最後（表示用 GPU のタイル `local_ev_` も待たせてから）で記録する（`record_decode_ready`）。

## 7. デコードの先行（decode_thread）

- 順方向の通常再生だけ、GUI の未受け取り（`inflight_frames`）が 2 未満なら受け取りを待たずに次の刻みでもデコードする。
  1:1 だと 1 コマの周期が「デコード待ち ＋ GUI の受け取り・描画」の合計になる。
- エンコード中・一時停止・逆再生・コマ送り・シークでは使わない。一時停止はボタンより 1 コマ後で止まる（コマ送り・戻しは正確に ±1）。
- `CUDA_VIEWER_DECODE_AHEAD=0`（または 1）で従来の 1:1。
- 8 タイルでは効く（外すと 205 → 150fps）。4 タイルでは NVDEC 律速で差が見えなかった。

## 8. 性能 HUD の値

| 行 | 意味 |
|---|---|
| デコード（待ち） | `get_decode_image` の実時間。内訳（▸）: 読み込み / GPU ごとの送る・受取 / 転送の発行 / その他（CPU 実時間）。待ちはほぼ「送る」に出る |
| 圧縮 | 別 GPU の前処理＋圧縮（GPU ごとの最大） |
| 転送 | 圧縮なし: D2H。圧縮あり: メインメモリへの書き出し（枠の空き待ちは除く） |
| 展開 | 表示用 GPU の取り込み〜差分戻し（GPU ごとの合計）。内訳（▸）: 取り込み / nvCOMP 展開 / 差分戻し・書き込み |
| 圧縮率 · 実効 GB/s | 元 ÷ 圧縮後、元の大きさ ÷ 圧縮と送出の遅い方 |
| NVDEC | GPU ごとの使用率（NVML と PDH の VideoDecode の大きい方） |
| デコード上限（目安） | GPU ごとに fps ÷ 動いている基の使用率 の最小。使用率は全基の平均なので、タイル数 < 基数の GPU は「使用率 × 基数 ÷ タイル数」に直す。基数は `cuvidGetDecoderCaps` の `nNumNVDECs`（取れなければ 1） |

- GPU の時間はストリーム上の経過時間なので、並行して動く処理と SM を取り合った時間も含む（カーネルが速くなっても数字に出にくい）。
- HUD は安定して順方向に再生している間だけ更新する（それ以外は直前の値で止めて「停止中 ·」）。

## 9. 設定・環境変数

| 項目 | 場所 |
|---|---|
| 重み | `GPUInfo::decode_weight`、ini の GPU グループ `decode_weight` |
| 転送の圧縮 | `g_AppSettings.decode_xfer_codec`（`none` / `ans` / `cascaded` / `bitcomp`、既定 `ans`）、ini `Decode/xfer_codec` |
| 差分 | `g_AppSettings.decode_xfer_delta`（既定 true）、ini `Decode/xfer_delta` |
| `CUDA_VIEWER_DECODE_AHEAD` | 0 / 1 でデコードの先行を切る |

## 10. 実測

- RTX 4080 SUPER 単体（16K 8bit 30fps）: NVDEC 80%、2 倍速で 37fps 頭打ち。
- RTX 4080 SUPER ＋ RTX 2070（Gen3 x4）、4 タイル: 3:1 で約 48fps、2:2 は 2070 の NVDEC が張り付いて約 42fps（圧縮あり）。
  圧縮なしの 2:2 は転送（約 3GB/s）で約 30fps。
- RTX 5000 Ada ×4（Gen4 x16）、16K 8bit 60fps の 4 倍速（目標 240fps）:
  - 4 タイル: 約 120fps（NVDEC 2 基のうち 1 基しか使わない）。
  - 8 タイル・ANS・差分あり: 約 205〜215fps。圧縮なしは転送 約 5.8ms で約 110fps。
  - 残る上限は NVDEC（GPU によって 80〜93%）・表示用 GPU の総負荷・デコードと描画のやり取りがほぼ同時に効いていて、単独のネックは無い。

### 試して効果がなく戻したもの

- 受け渡し枠 3 → 6、作業領域 2 → 3 組（待ちは変わらず）
- 展開を別 GPU ごとのストリームに分ける、16 バイト単位の差分戻し、取り込みの見込みを縮める（体感やや遅い）
- 差分戻しを丸ごと飛ばした上限測定 → fps 変わらず（差分戻しを変換カーネルへ統合する案は不採用）

## 11. 今後の候補

- タイルの割り振りをビットレート（デコードの重さの目安）で均す（NVDEC の偏り対策。計算上 約 235fps）
- デコード設定で「受け持ちタイル数 < NVDEC の基数」の GPU に注意を出す
- 高 fps 再生のときだけ先行 2 コマ（リング 8 段が必要。等倍では音ズレが目立つので条件付きで）
- 前処理を 2 次元の予測にして圧縮率を上げる（取り込みが軽くなる）
