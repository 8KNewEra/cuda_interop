#ifndef __GLOBAL___H
#define __GLOBAL___H

#include "qobject.h"
#include <string>
#include <cuda_runtime.h>

#define STATE_NOT_ENCODE 0
#define STATE_ENCODE_READY 1
#define STATE_ENCODING 2

// エンコード設定の構造体
struct EncodeSettings {
    std::string Save_Path = "D:/test1.mp4";

    std::string Codec{};
    std::string rc_mode{};
    std::string preset{};
    std::string tune{};
    std::string split_encode_mode{};
    std::string pass_mode{};
    int encode_tile = 1;
    int width_tile = 1;
    int height_tile = 1;

    int target_bit_rate{};
    int max_bit_rate{};
    int cq{};

    int gop_size = 60;
    int b_frames{};
    int save_fps{};
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
    std::string Codec = "av1";
    int max_framesNo=10;
    int pts_per_frame=1000;
    int current_frameNo=0;
    double fps = 30;
    int bitdepth=8;
    int width=3840;
    int height=2160;
    int width_scale=1;
    int height_scale=1;
    QString decode_mode="";
    bool audio = false;
    int audio_channels=1;
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
    unsigned int hist_r[1024];
    unsigned int hist_g[1024];
    unsigned int hist_b[1024];
    unsigned int max_r;
    unsigned int max_g;
    unsigned int max_b;
};

//デバイス情報
extern cudaDeviceProp g_prop;

//RGBレイアウト
enum RGBLayout {
    RGB = 0,
    BGR = 1,
    GBR = 2,
    RBG = 3,
    BRG = 4,
    GRB = 5
};

#endif // __GLOBAL___H
