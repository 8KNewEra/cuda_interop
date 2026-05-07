#include "src/info/logfile_control.h"

logfile_control::logfile_control(){
    load_inifile();
}

logfile_control::~logfile_control(){
    save_inifile();
}

void logfile_control::load_inifile()
{
    // setting.ini が無い場合は作成
    if (!QFile::exists("setting.ini")) {
        save_inifile();   // ← 初期値を書き込んで ini を作成
    }
    QSettings settings("setting.ini", QSettings::IniFormat);

    g_AppSettings.decode_path = settings.value("Path/decode_path").toString();
    g_AppSettings.videoInfo_flag = settings.value("Info/videoInfo_flag").toBool();
    g_AppSettings.histgram_flag = settings.value("Info/histgram_flag").toBool();
    g_AppSettings.video_speed_ratio = settings.value("Playback/video_speed_ratio").toDouble();
    g_AppSettings.audio_volume = settings.value("Playback/audio_volume").toInt();
    g_AppSettings.frame_jump_mode = settings.value("Playback/frame_jump_mode").toInt();
    g_AppSettings.jumpValueSecond = settings.value("Playback/jumpValueSecond").toInt();
    g_AppSettings.jumpValueFrame = settings.value("Playback/jumpValueFrame").toInt();
    g_AppSettings.jumpValueFrameNo = settings.value("Playback/jumpValueFrameNo").toInt();
    g_AppSettings.audio_low_laytency_flag = settings.value("Playback/audio_low_laytency_flag").toBool();
    g_AppSettings.sobelfilterEnabled = settings.value("Filter/sobelfilterEnabled").toBool();
    g_AppSettings.gaussianfilterEnabled = settings.value("Filter/gaussianfilterEnabled").toBool();
    g_AppSettings.averagingfilterEnabled = settings.value("Filter/averagingfilterEnabled").toBool();

    encodeSettings.encode_path = settings.value("Encodesetting/encode_path").toString();
    encodeSettings.codec = settings.value("Encodesetting/codec").toString();
    encodeSettings.preset = settings.value("Encodesetting/preset").toString();
    encodeSettings.tune = settings.value("Encodesetting/tune").toString();
    encodeSettings.b_frames = settings.value("Encodesetting/b_frames").toInt();
    encodeSettings.gop_size = settings.value("Encodesetting/gop_size").toInt();
    encodeSettings.encode_tile = settings.value("Encodesetting/encode_tile").toInt();
    encodeSettings.split_encode_mode = settings.value("Encodesetting/split_encode_mode").toString();
    encodeSettings.pass_mode = settings.value("Encodesetting/pass_mode").toString();
    encodeSettings.rc_mode = settings.value("Encodesetting/rc_mode").toString();
    encodeSettings.target_bit_rate = settings.value("Encodesetting/target_bit_rate").toInt();
    encodeSettings.max_bit_rate = settings.value("Encodesetting/max_bit_rate").toInt();
    encodeSettings.cq = settings.value("Encodesetting/cq").toInt();

    // ==========================
    // GPUInfo 読み込み（追記）
    // ==========================
    settings.beginGroup("GPUInfo");
    int count = settings.value("count", 0).toInt();
    g_GPUInfo.clear();
    g_GPUInfo.resize(count);
    for (int i = 0; i < count; i++)
    {
        settings.beginGroup(QString("GPU_%1").arg(i));

        g_GPUInfo[i].deviceName = settings.value("deviceName", "").toString();
        g_GPUInfo[i].deviceID = settings.value("deviceID", 0).toInt();
        g_GPUInfo[i].openglEnable = settings.value("openglEnable", false).toBool();
        g_GPUInfo[i].CC_major = settings.value("CC_major", 0).toInt();
        g_GPUInfo[i].CC_minor = settings.value("CC_minor", 0).toInt();
        g_GPUInfo[i].pciDomain = settings.value("pciDomain", 0).toInt();
        g_GPUInfo[i].pciBus = settings.value("pciBus", 0).toInt();
        g_GPUInfo[i].pciDevice = settings.value("pciDevice", 0).toInt();
        g_GPUInfo[i].Max_Memory_Usage = settings.value("Max_Memory_Usage", 0).toInt();
        g_GPUInfo[i].Memory_Usage = settings.value("Memory_Usage", 0).toInt();
        g_GPUInfo[i].GPU_Usage = settings.value("GPU_Usage", 0).toInt();
        g_GPUInfo[i].tile_weight = settings.value("tile_weight", 10).toInt();

        settings.endGroup();
    }
    settings.endGroup();
}

void logfile_control::save_inifile(){
    QSettings settings("setting.ini", QSettings::IniFormat);
    settings.setValue("Path/decode_path", g_AppSettings.decode_path);
    settings.setValue("Info//videoInfo_flag", g_AppSettings.videoInfo_flag);
    settings.setValue("Info/histgram_flag", g_AppSettings.histgram_flag);
    settings.setValue("Playback/video_speed_ratio", g_AppSettings.video_speed_ratio);
    settings.setValue("Playback/audio_volume", g_AppSettings.audio_volume);
    settings.setValue("Playback/frame_jump_mode", g_AppSettings.frame_jump_mode);
    settings.setValue("Playback/jumpValueSecond", g_AppSettings.jumpValueSecond);
    settings.setValue("Playback/jumpValueFrame", g_AppSettings.jumpValueFrame);
    settings.setValue("Playback/jumpValueFrameNo", g_AppSettings.jumpValueFrameNo);
    settings.setValue("Playback/audio_low_laytency_flag", g_AppSettings.audio_low_laytency_flag);
    settings.setValue("Filter/sobelfilterEnabled", g_AppSettings.sobelfilterEnabled);
    settings.setValue("Filter/gaussianfilterEnabled", g_AppSettings.gaussianfilterEnabled);
    settings.setValue("Filter/averagingfilterEnabled", g_AppSettings.averagingfilterEnabled);

    settings.setValue("Encodesetting/encode_path", encodeSettings.encode_path);
    settings.setValue("Encodesetting/codec", encodeSettings.codec);
    settings.setValue("Encodesetting/preset", encodeSettings.preset);
    settings.setValue("Encodesetting/tune", encodeSettings.tune);
    settings.setValue("Encodesetting/b_frames", encodeSettings.b_frames);
    settings.setValue("Encodesetting/gop_size", encodeSettings.gop_size);
    settings.setValue("Encodesetting/encode_tile", encodeSettings.encode_tile);
    settings.setValue("Encodesetting/split_encode_mode", encodeSettings.split_encode_mode);
    settings.setValue("Encodesetting/pass_mode", encodeSettings.pass_mode);
    settings.setValue("Encodesetting/rc_mode", encodeSettings.rc_mode);
    settings.setValue("Encodesetting/target_bit_rate", encodeSettings.target_bit_rate);
    settings.setValue("Encodesetting/max_bit_rate", encodeSettings.max_bit_rate);
    settings.setValue("Encodesetting/cq", encodeSettings.cq);

    // ==========================
    // GPUInfo 保存（追記）
    // ==========================
    settings.beginGroup("GPUInfo");
    settings.remove(""); // GPUInfo以下を全削除（古いゴミ対策）
    settings.setValue("count", (int)g_GPUInfo.size());
    for(int i = 0; i < (int)g_GPUInfo.size(); i++)
    {
        settings.beginGroup(QString("GPU_%1").arg(i));

        settings.setValue("deviceName", g_GPUInfo[i].deviceName);
        settings.setValue("deviceID", g_GPUInfo[i].deviceID);
        settings.setValue("openglEnable", g_GPUInfo[i].openglEnable);
        settings.setValue("CC_major", g_GPUInfo[i].CC_major);
        settings.setValue("CC_minor", g_GPUInfo[i].CC_minor);
        settings.setValue("pciDomain", g_GPUInfo[i].pciDomain);
        settings.setValue("pciBus", g_GPUInfo[i].pciBus);
        settings.setValue("pciDevice", g_GPUInfo[i].pciDevice);
        settings.setValue("Max_Memory_Usage", g_GPUInfo[i].Max_Memory_Usage);
        settings.setValue("Memory_Usage", g_GPUInfo[i].Memory_Usage);
        settings.setValue("GPU_Usage", g_GPUInfo[i].GPU_Usage);
        settings.setValue("tile_weight", g_GPUInfo[i].tile_weight);

        settings.endGroup();
    }

    settings.endGroup();
    settings.sync();
}
