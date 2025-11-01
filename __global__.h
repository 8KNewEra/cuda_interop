#ifndef __GLOBAL___H
#define __GLOBAL___H

#include <string> // std::string を使うために必要

// エンコード設定の構造体
struct EncodeSettings {
    std::string Save_Path = "D:/test1.mp4";

    std::string Codec{};
    std::string rc_mode{};
    std::string preset{};
    std::string tune{};
    std::string split_encode_mode{};
    std::string pass_mode{};

    int target_bit_rate{};
    int max_bit_rate{};
    int crf{};

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
    double max_frames_pts = 0;
    int max_framesNo=0;
    int pts_per_frame=0;
    int current_frameNo=0;
    double fps = 30;
    int width=0;
    int height=0;
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

#endif // __GLOBAL___H
