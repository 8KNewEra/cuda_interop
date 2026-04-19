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
}
