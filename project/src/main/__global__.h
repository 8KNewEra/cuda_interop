#ifndef __GLOBAL___H
#define __GLOBAL___H

#include "qcoreapplication.h"
#include "qobject.h"
#include <QFileDialog>
#include <string>
#include <cuda_runtime.h>
#define STATE_NOT_ENCODE 0
#define STATE_ENCODE_READY 1
#define STATE_ENCODING 2

extern "C" {
    #include <libavutil/samplefmt.h>
}

// アプリ設定を保持する構造体
struct AppSettings
{
    // Path
    QString decode_path = QDir::homePath();

    // Info
    bool videoInfo_flag = false;
    bool histgram_flag  = false;

    // Playback
    double video_speed_ratio = 1.0;
    int audio_volume = 50;
    int frame_jump_mode = 0;
    int jumpValueSecond = 1;
    int jumpValueFrame = 1;
    int jumpValueFrameNo = 0;

    // Filter
    bool sobelfilterEnabled = false;
    bool gaussianfilterEnabled = false;
    bool averagingfilterEnabled = false;
};

// エンコード設定の構造体
struct EncodeSettings {
    // Encode setting
    QString encode_path = QCoreApplication::applicationDirPath()+"/output.mp4";
    QString codec  = "h264_nvenc";
    QString preset = "p4";
    QString tune   = "default";

    double save_fps{};
    int b_frames = 0;
    int gop_size = 15;

    QString split_encode_mode = "0";
    QString pass_mode = "1pass";
    QString rc_mode   = "cbr";

    int target_bit_rate = 100000000;
    int max_bit_rate    = 1000000000;
    int cq = 23;

    int encode_tile = 1;
    int width_tile = 1;
    int height_tile = 1;
    std::vector<int> tile_gpu_map;
};

// ★書き込みを許可したいクラスをここで「前方宣言」
class encode_setting;

// 設定を管理するシングルトンクラス
class EncodeSettingsManager
{
public:
    // 1. インスタンスを取得する唯一の口
    static EncodeSettingsManager& getInstance()
    {
        static EncodeSettingsManager s_instance;
        return s_instance;
    }

    // 2.【読み取り用】
    //    誰でも呼べる (const参照)
    const EncodeSettings& getSettings() const
    {
        return m_settings;
    }

private:
    // 3. ★書き込みクラスとして 'encode_setting' を指定
    friend class logfile_control;
    friend class encode_setting;

    // 4.【書き込み用】
    //    'encode_setting' クラスだけが呼び出せる
    EncodeSettings& getSettingsNonConst()
    {
        return m_settings;
    }

    // 5. シングルトンのための設定
    EncodeSettingsManager() {}
    ~EncodeSettingsManager() {}
    EncodeSettingsManager(const EncodeSettingsManager&) = delete;
    void operator=(const EncodeSettingsManager&) = delete;

    // 6. 設定の実体を private で保持
    EncodeSettings m_settings;
};

// エンコード設定の構造体
struct DecodeInfo {
    std::string Path = "D:/test2.mp4";
    std::string Name = "";
    bool video_open_flag = false;

    //映像メタデータ
    std::string Codec = "av1";
    int max_framesNo=10;
    int end_range_framesNo=10;
    int start_range_framesNo=0;
    int pts_per_frame=1000;
    double fps = 30;
    int bitdepth=8;
    int width=3840;
    int height=2160;
    int width_scale=1;
    int height_scale=1;
    QString decode_mode="";

    //時間
    int max_hour = 0;
    int max_minute = 0;
    int max_second = 0;

    //音声
    bool audio = false;
    int audio_channels=1;
    int in_sample_rate  = 0;
    int out_sample_rate = 0;
    AVSampleFormat in_format  = AV_SAMPLE_FMT_NONE;

    // ★ デコード後に配る PCM の形式（再生とエンコード入力の両方）。
    //   float のまま配ること。S16 へ先に落とすと、0dBFS を超える素材
    //   （PCM 32bit float など）がその時点で飽和し、後から音量を下げても
    //   歪みが残る。音量は各出口が float のうちに掛ける。
    AVSampleFormat out_format = AV_SAMPLE_FMT_FLT;  // インターリーブ float
};

// ★書き込みを許可したいクラスをここで「前方宣言」
class decode_thread;

// 設定を管理するシングルトンクラス
class DecodeInfoManager
{
public:
    // 1. インスタンスを取得する唯一の口
    static DecodeInfoManager& getInstance()
    {
        static DecodeInfoManager s_instance;
        return s_instance;
    }

    // 2.【読み取り用】
    //    誰でも呼べる (const参照)
    const DecodeInfo& getSettings() const
    {
        return m_settings;
    }

private:
    // 3. ★書き込みクラスとして 'encode_setting' を指定
    friend class decode_thread;

    // 4.【書き込み用】
    //    'encode_setting' クラスだけが呼び出せる
    DecodeInfo& getSettingsNonConst()
    {
        return m_settings;
    }

    // 5. シングルトンのための設定
    DecodeInfoManager() {}
    ~DecodeInfoManager() {}
    DecodeInfoManager(const DecodeInfoManager&) = delete;
    void operator=(const DecodeInfoManager&) = delete;

    // 6. 設定の実体を private で保持
    DecodeInfo m_settings;
};

// ==============================================================
// デコード結果の受け渡し形式
//   どちらもデコード結果を直接指すゼロコピー。中間バッファは持たない。
//
//   SRC_NV12_TILE : NVDEC が出した NV12 タイル（nvgpudecode）
//   SRC_YUV_PLANE : CPUデコードをGPUへ上げた YUV プレーン（cpudecode/avidecode）
//
//   いずれも GUIスレッドが surface object でGLテクスチャへ直接書く。
// ==============================================================
enum VideoFrameSrc {
    SRC_NONE      = 0,   // 未設定（デコーダが埋め忘れている）
    SRC_NV12_TILE = 1,
    SRC_YUV_PLANE = 2
};

// SRC_YUV_PLANE のときの画素フォーマット
//   AVPixelFormat を直接持たせると glwidget が ffmpeg に依存するので、
//   必要な種類だけを自前の enum に落としている。
enum YuvPlaneFormat {
    YUVFMT_NONE = 0,
    YUVFMT_YUV420P_8,
    YUVFMT_YUV420P_10,
    YUVFMT_YUV422P_8,
    YUVFMT_YUV422P_10,
    YUVFMT_UYVY422_8,    // packed。plane[0] のみ使用
    YUVFMT_GBRP_10       // plane[0]=G, plane[1]=B, plane[2]=R
};

//デコードフレーム
struct VideoFrame {
    static const int MAX_TILES = 8;

    //映像の受け渡し形式
    int src_mode = SRC_NONE;

    // --- SRC_NV12_TILE のとき使用 ---
    //   NVDEC が出力した NV12 タイルを直接指す（コピーしない）
    //   有効期間はデコード側のリング（vd[i].hw_frames）が一周するまで
    int      tile_count = 0;                  // 1 / 2 / 4 / 8
    int      tile_w     = 0;                  // 1タイルの幅  (= VideoInfo.width)
    int      tile_h     = 0;                  // 1タイルの高さ(= VideoInfo.height)
    uint8_t* d_y     [MAX_TILES] = {};
    size_t   y_pitch [MAX_TILES] = {};
    uint8_t* d_uv    [MAX_TILES] = {};
    size_t   uv_pitch[MAX_TILES] = {};

    // --- SRC_YUV_PLANE のとき使用 ---
    //   CPUデコード結果をGPUへ上げただけのプレーンを直接指す（RGBA化しない）
    //   有効期間はデコード側のプレーンリングが一周するまで
    int      plane_format = YUVFMT_NONE;
    int      plane_is_be  = 0;          // 10bit BE のときのみ 1
    uint8_t* d_plane    [3] = {};
    size_t   plane_pitch[3] = {};

    int FrameNo=0;

    // ★ NV12 の GPU 転送完了イベント。
    //   cuvid は出力を cuMemcpy2DAsync で積むだけで同期せずに戻るため
    //   (libavcodec/cuviddec.c)、受け取った直後の d_y/d_uv はまだ転送中。
    //   GUI 側は結合カーネルの前に cudaStreamWaitEvent で待つこと。
    //   nullptr なら待ち不要（CPUデコード経路）。
    cudaEvent_t decode_ready = nullptr;

    //時間
    int hour = 0;
    int minute = 0;
    int second = 0;

    //音声
    QVector<QByteArray> audio_pcm{};
    QVector<int> audio_pts{};
};

//UI表示CPUダウンロード用
struct HistStats {
    int min_r, max_r;
    int min_g, max_g;
    int min_b, max_b;
    double avg_r, avg_g, avg_b;
    int max_y_axis;
};

//GPU内で保持する
struct HistData {
    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

//RGBレイアウト
enum RGBLayout {
    RGB = 0,
    BGR = 1,
    GBR = 2,
    RBG = 3,
    BRG = 4,
    GRB = 5
};

//フレームジャンプモード
enum e_jump_mode{
    JUMP_MODE_SECOND = 0,
    JUMP_MODE_FRAME = 1,
    JUMP_MODE_TARGETFRAME = 2
};

//GPU情報を持つ構造体
struct GPUInfo {
    QString deviceName ="";
    int deviceID = 0;
    bool openglEnable = false;
    int CC_major = 0;
    int CC_minor = 0;
    int pciDomain = 0;
    int pciBus = 0;
    int pciDevice = 0;
    int Max_Memory_Usage = 0;
    int Memory_Usage = 0;
    int GPU_Usage = 0;
    //エンコード設定用
    int tile_weight = 10;
    int backup_tile_weight = -1;
    bool av1_forced_disabled = false;
};

extern AppSettings g_AppSettings;
extern std::vector<GPUInfo> g_GPUInfo;
extern int g_openglDeviceID;

#endif // __GLOBAL___H
