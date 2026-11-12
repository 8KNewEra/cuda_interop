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
        save_inifile();   // 初期値を書き込んで ini を作成
    }
    QSettings settings("setting.ini", QSettings::IniFormat);

    // 既定値は必ず渡す。無いとキーが無いとき 0 / 空文字になり、項目を増やすたびに
    //   既存の ini のユーザーがその項目 0 で起動する（video_speed_ratio 0 = 再生速度 0 倍）
    g_AppSettings.decode_path = settings.value("Path/decode_path", g_AppSettings.decode_path).toString();
    g_AppSettings.saveScreenshot_folder = settings.value("Path/saveScreenshot_folder", g_AppSettings.saveScreenshot_folder).toString();
    g_AppSettings.videoInfo_flag = settings.value("Info/videoInfo_flag", g_AppSettings.videoInfo_flag).toBool();
    g_AppSettings.histgram_flag = settings.value("Info/histgram_flag", g_AppSettings.histgram_flag).toBool();
    g_AppSettings.video_speed_ratio = settings.value("Playback/video_speed_ratio", g_AppSettings.video_speed_ratio).toDouble();
    g_AppSettings.audio_volume = settings.value("Playback/audio_volume", g_AppSettings.audio_volume).toInt();
    g_AppSettings.frame_jump_mode = settings.value("Playback/frame_jump_mode", g_AppSettings.frame_jump_mode).toInt();
    g_AppSettings.jumpValueSecond = settings.value("Playback/jumpValueSecond", g_AppSettings.jumpValueSecond).toInt();
    g_AppSettings.jumpValueFrame = settings.value("Playback/jumpValueFrame", g_AppSettings.jumpValueFrame).toInt();
    g_AppSettings.jumpValueFrameNo = settings.value("Playback/jumpValueFrameNo", g_AppSettings.jumpValueFrameNo).toInt();
    g_AppSettings.sobelfilterEnabled = settings.value("Filter/sobelfilterEnabled", g_AppSettings.sobelfilterEnabled).toBool();
    g_AppSettings.gaussianfilterEnabled = settings.value("Filter/gaussianfilterEnabled", g_AppSettings.gaussianfilterEnabled).toBool();
    g_AppSettings.averagingfilterEnabled = settings.value("Filter/averagingfilterEnabled", g_AppSettings.averagingfilterEnabled).toBool();
    g_AppSettings.gaussian_kernel_size = settings.value("Filter/gaussian_kernel_size", g_AppSettings.gaussian_kernel_size).toInt();
    g_AppSettings.gaussian_sigma100 = settings.value("Filter/gaussian_sigma100", g_AppSettings.gaussian_sigma100).toInt();
    g_AppSettings.gaussian_normalize = settings.value("Filter/gaussian_normalize", g_AppSettings.gaussian_normalize).toBool();
    g_AppSettings.averaging_kernel_size = settings.value("Filter/averaging_kernel_size", g_AppSettings.averaging_kernel_size).toInt();
    g_AppSettings.averaging_normalize = settings.value("Filter/averaging_normalize", g_AppSettings.averaging_normalize).toBool();
    g_AppSettings.sobel_mode = settings.value("Filter/sobel_mode", g_AppSettings.sobel_mode).toInt();
    g_AppSettings.peaking_threshold = settings.value("Filter/peaking_threshold", g_AppSettings.peaking_threshold).toInt();
    g_AppSettings.peaking_color = settings.value("Filter/peaking_color", g_AppSettings.peaking_color).toInt();
    g_AppSettings.zebraEnabled = settings.value("Filter/zebraEnabled", g_AppSettings.zebraEnabled).toBool();
    g_AppSettings.eyedropperEnabled = settings.value("Filter/eyedropperEnabled", g_AppSettings.eyedropperEnabled).toBool();
    g_AppSettings.eyedropper_mode = settings.value("Filter/eyedropper_mode", g_AppSettings.eyedropper_mode).toInt();
    g_AppSettings.seamEnabled = settings.value("Filter/seamEnabled", g_AppSettings.seamEnabled).toBool();
    g_AppSettings.seam_grid = settings.value("Filter/seam_grid", g_AppSettings.seam_grid).toBool();
    g_AppSettings.seam_highlight = settings.value("Filter/seam_highlight", g_AppSettings.seam_highlight).toBool();
    g_AppSettings.seam_threshold = settings.value("Filter/seam_threshold", g_AppSettings.seam_threshold).toInt();
    g_AppSettings.seam_grid_color = settings.value("Filter/seam_grid_color", g_AppSettings.seam_grid_color).toInt();
    g_AppSettings.seam_highlight_color = settings.value("Filter/seam_highlight_color", g_AppSettings.seam_highlight_color).toInt();
    g_AppSettings.compareEnabled = settings.value("Filter/compareEnabled", g_AppSettings.compareEnabled).toBool();
    g_AppSettings.compare_dir = settings.value("Filter/compare_dir", g_AppSettings.compare_dir).toInt();
    // 位置は 0.1% 単位（キー compare_pos10）。旧キー compare_pos（% 単位）があれば 10 倍して引き継ぐ
    if (settings.contains("Filter/compare_pos10"))
        g_AppSettings.compare_pos10 = settings.value("Filter/compare_pos10").toInt();
    else if (settings.contains("Filter/compare_pos"))
        g_AppSettings.compare_pos10 = settings.value("Filter/compare_pos").toInt() * 10;
    // トーンカーブ（読み込み時の正規化は ToneCurve::parse が行う）
    g_AppSettings.curveEnabled = settings.value("Filter/curveEnabled", g_AppSettings.curveEnabled).toBool();
    {
        static const char* kCurveKey[4] = { "Filter/curve_master", "Filter/curve_r", "Filter/curve_g", "Filter/curve_b" };
        for (int i = 0; i < 4; ++i)
            g_AppSettings.curve_pts[i] = settings.value(kCurveKey[i], g_AppSettings.curve_pts[i]).toString();
    }
    g_AppSettings.gaussian_mode = settings.value("Filter/gaussian_mode", g_AppSettings.gaussian_mode).toInt();
    g_AppSettings.sharpen_amount = settings.value("Filter/sharpen_amount", g_AppSettings.sharpen_amount).toInt();
    g_AppSettings.channelEnabled = settings.value("Filter/channelEnabled", g_AppSettings.channelEnabled).toBool();
    g_AppSettings.channel_mode = settings.value("Filter/channel_mode", g_AppSettings.channel_mode).toInt();
    g_AppSettings.colorEnabled = settings.value("Filter/colorEnabled", g_AppSettings.colorEnabled).toBool();
    g_AppSettings.color_brightness = settings.value("Filter/color_brightness", g_AppSettings.color_brightness).toInt();
    g_AppSettings.color_contrast = settings.value("Filter/color_contrast", g_AppSettings.color_contrast).toInt();
    g_AppSettings.color_shadows = settings.value("Filter/color_shadows", g_AppSettings.color_shadows).toInt();
    g_AppSettings.color_highlights = settings.value("Filter/color_highlights", g_AppSettings.color_highlights).toInt();
    g_AppSettings.color_gamma100 = settings.value("Filter/color_gamma100", g_AppSettings.color_gamma100).toInt();
    g_AppSettings.color_saturation = settings.value("Filter/color_saturation", g_AppSettings.color_saturation).toInt();
    g_AppSettings.color_temperature = settings.value("Filter/color_temperature", g_AppSettings.color_temperature).toInt();
    g_AppSettings.color_tint = settings.value("Filter/color_tint", g_AppSettings.color_tint).toInt();
    // 色補正（R/G/B）。旧 ini で彩度を変えていた場合は、効き続けるよう色補正を ON で始める
    if (settings.contains("Filter/rgbEnabled"))
        g_AppSettings.rgbEnabled = settings.value("Filter/rgbEnabled").toBool();
    else
        g_AppSettings.rgbEnabled = g_AppSettings.colorEnabled && g_AppSettings.color_saturation != 100;
    for (int i = 0; i < 3; ++i) {
        const char ch = "RGB"[i];
        g_AppSettings.rgb_gain[i]     = settings.value(QString("Filter/rgb_gain_%1").arg(ch),     g_AppSettings.rgb_gain[i]).toInt();
        g_AppSettings.rgb_offset[i]   = settings.value(QString("Filter/rgb_offset_%1").arg(ch),   g_AppSettings.rgb_offset[i]).toInt();
        g_AppSettings.rgb_gamma100[i] = settings.value(QString("Filter/rgb_gamma100_%1").arg(ch), g_AppSettings.rgb_gamma100[i]).toInt();
    }
    g_AppSettings.zebra_high_on = settings.value("Filter/zebra_high_on", g_AppSettings.zebra_high_on).toBool();
    g_AppSettings.zebra_high_level = settings.value("Filter/zebra_high_level", g_AppSettings.zebra_high_level).toInt();
    g_AppSettings.zebra_low_on = settings.value("Filter/zebra_low_on", g_AppSettings.zebra_low_on).toBool();
    g_AppSettings.zebra_low_level = settings.value("Filter/zebra_low_level", g_AppSettings.zebra_low_level).toInt();
    g_AppSettings.screenshot_format = settings.value("Screenshot/screenshot_format", g_AppSettings.screenshot_format).toString();
    g_AppSettings.screenshot_mode = settings.value("Screenshot/screenshot_mode", g_AppSettings.screenshot_mode).toString();
    g_AppSettings.screenshot_tiff_compression = settings.value("Screenshot/screenshot_tiff_compression", g_AppSettings.screenshot_tiff_compression).toString();
    g_AppSettings.screenshot_jpeg_quality = settings.value("Screenshot/screenshot_jpeg_quality", g_AppSettings.screenshot_jpeg_quality).toInt();
    g_AppSettings.screenshot_jpeg_subsampling = settings.value("Screenshot/screenshot_jpeg_subsampling", g_AppSettings.screenshot_jpeg_subsampling).toString();

    encodeSettings.encode_path = settings.value("Encodesetting/encode_path", encodeSettings.encode_path).toString();
    encodeSettings.container = settings.value("Encodesetting/container", encodeSettings.container).toString();
    encodeSettings.separate_files = settings.value("Encodesetting/separate_files", encodeSettings.separate_files).toBool();
    encodeSettings.audio_enable = settings.value("Encodesetting/audio_enable", encodeSettings.audio_enable).toBool();
    encodeSettings.audio_codec = settings.value("Encodesetting/audio_codec", encodeSettings.audio_codec).toString();
    encodeSettings.audio_gain_db = settings.value("Encodesetting/audio_gain_db", encodeSettings.audio_gain_db).toDouble();
    encodeSettings.audio_bitrate = settings.value("Encodesetting/audio_bitrate", encodeSettings.audio_bitrate).toInt();
    encodeSettings.codec = settings.value("Encodesetting/codec", encodeSettings.codec).toString();
    encodeSettings.preset = settings.value("Encodesetting/preset", encodeSettings.preset).toString();
    encodeSettings.tune = settings.value("Encodesetting/tune", encodeSettings.tune).toString();
    encodeSettings.b_frames = settings.value("Encodesetting/b_frames", encodeSettings.b_frames).toInt();
    encodeSettings.gop_size = settings.value("Encodesetting/gop_size", encodeSettings.gop_size).toInt();
    encodeSettings.encode_tile = settings.value("Encodesetting/encode_tile", encodeSettings.encode_tile).toInt();
    encodeSettings.split_encode_mode = settings.value("Encodesetting/split_encode_mode", encodeSettings.split_encode_mode).toString();
    encodeSettings.pass_mode = settings.value("Encodesetting/pass_mode", encodeSettings.pass_mode).toString();
    encodeSettings.rc_mode = settings.value("Encodesetting/rc_mode", encodeSettings.rc_mode).toString();
    encodeSettings.target_bit_rate = settings.value("Encodesetting/target_bit_rate", encodeSettings.target_bit_rate).toInt();
    encodeSettings.max_bit_rate = settings.value("Encodesetting/max_bit_rate", encodeSettings.max_bit_rate).toInt();
    encodeSettings.cq = settings.value("Encodesetting/cq", encodeSettings.cq).toInt();
    encodeSettings.bitrate_basis = settings.value("Encodesetting/bitrate_basis", encodeSettings.bitrate_basis).toString();

    // 知らない値が入っていたら既定へ倒す（手で ini を編集した場合の保険）
    if (encodeSettings.bitrate_basis != "total" && encodeSettings.bitrate_basis != "stream")
        encodeSettings.bitrate_basis = "stream";
    if (encodeSettings.container != "mp4" && encodeSettings.container != "mkv" &&
        encodeSettings.container != "mov")
        encodeSettings.container = "mp4";

    // ==========================
    // GPUInfo 読み込み
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
    settings.setValue("Path/saveScreenshot_folder", g_AppSettings.saveScreenshot_folder);
    settings.setValue("Info/videoInfo_flag", g_AppSettings.videoInfo_flag);
    settings.setValue("Info/histgram_flag", g_AppSettings.histgram_flag);
    settings.setValue("Playback/video_speed_ratio", g_AppSettings.video_speed_ratio);
    settings.setValue("Playback/audio_volume", g_AppSettings.audio_volume);
    settings.setValue("Playback/frame_jump_mode", g_AppSettings.frame_jump_mode);
    settings.setValue("Playback/jumpValueSecond", g_AppSettings.jumpValueSecond);
    settings.setValue("Playback/jumpValueFrame", g_AppSettings.jumpValueFrame);
    settings.setValue("Playback/jumpValueFrameNo", g_AppSettings.jumpValueFrameNo);
    settings.setValue("Filter/sobelfilterEnabled", g_AppSettings.sobelfilterEnabled);
    settings.setValue("Filter/gaussianfilterEnabled", g_AppSettings.gaussianfilterEnabled);
    settings.setValue("Filter/averagingfilterEnabled", g_AppSettings.averagingfilterEnabled);
    settings.setValue("Filter/gaussian_kernel_size", g_AppSettings.gaussian_kernel_size);
    settings.setValue("Filter/gaussian_sigma100", g_AppSettings.gaussian_sigma100);
    settings.setValue("Filter/gaussian_normalize", g_AppSettings.gaussian_normalize);
    settings.setValue("Filter/averaging_kernel_size", g_AppSettings.averaging_kernel_size);
    settings.setValue("Filter/averaging_normalize", g_AppSettings.averaging_normalize);
    settings.setValue("Filter/sobel_mode", g_AppSettings.sobel_mode);
    settings.setValue("Filter/peaking_threshold", g_AppSettings.peaking_threshold);
    settings.setValue("Filter/peaking_color", g_AppSettings.peaking_color);
    settings.setValue("Filter/zebraEnabled", g_AppSettings.zebraEnabled);
    settings.setValue("Filter/eyedropperEnabled", g_AppSettings.eyedropperEnabled);
    settings.setValue("Filter/eyedropper_mode", g_AppSettings.eyedropper_mode);
    settings.setValue("Filter/seamEnabled", g_AppSettings.seamEnabled);
    settings.setValue("Filter/seam_grid", g_AppSettings.seam_grid);
    settings.setValue("Filter/seam_highlight", g_AppSettings.seam_highlight);
    settings.setValue("Filter/seam_threshold", g_AppSettings.seam_threshold);
    settings.setValue("Filter/seam_grid_color", g_AppSettings.seam_grid_color);
    settings.setValue("Filter/seam_highlight_color", g_AppSettings.seam_highlight_color);
    settings.setValue("Filter/compareEnabled", g_AppSettings.compareEnabled);
    settings.setValue("Filter/compare_dir", g_AppSettings.compare_dir);
    settings.setValue("Filter/compare_pos10", g_AppSettings.compare_pos10);
    settings.remove("Filter/compare_pos");
    settings.setValue("Filter/curveEnabled", g_AppSettings.curveEnabled);
    {
        static const char* kCurveKey[4] = { "Filter/curve_master", "Filter/curve_r", "Filter/curve_g", "Filter/curve_b" };
        for (int i = 0; i < 4; ++i)
            settings.setValue(kCurveKey[i], g_AppSettings.curve_pts[i]);
    }
    settings.setValue("Filter/gaussian_mode", g_AppSettings.gaussian_mode);
    settings.setValue("Filter/sharpen_amount", g_AppSettings.sharpen_amount);
    settings.setValue("Filter/channelEnabled", g_AppSettings.channelEnabled);
    settings.setValue("Filter/channel_mode", g_AppSettings.channel_mode);
    settings.setValue("Filter/colorEnabled", g_AppSettings.colorEnabled);
    settings.setValue("Filter/color_brightness", g_AppSettings.color_brightness);
    settings.setValue("Filter/color_contrast", g_AppSettings.color_contrast);
    settings.setValue("Filter/color_shadows", g_AppSettings.color_shadows);
    settings.setValue("Filter/color_highlights", g_AppSettings.color_highlights);
    settings.setValue("Filter/color_gamma100", g_AppSettings.color_gamma100);
    settings.setValue("Filter/color_saturation", g_AppSettings.color_saturation);
    settings.setValue("Filter/color_temperature", g_AppSettings.color_temperature);
    settings.setValue("Filter/color_tint", g_AppSettings.color_tint);
    settings.setValue("Filter/rgbEnabled", g_AppSettings.rgbEnabled);
    for (int i = 0; i < 3; ++i) {
        const char ch = "RGB"[i];
        settings.setValue(QString("Filter/rgb_gain_%1").arg(ch),     g_AppSettings.rgb_gain[i]);
        settings.setValue(QString("Filter/rgb_offset_%1").arg(ch),   g_AppSettings.rgb_offset[i]);
        settings.setValue(QString("Filter/rgb_gamma100_%1").arg(ch), g_AppSettings.rgb_gamma100[i]);
    }
    settings.setValue("Filter/zebra_high_on", g_AppSettings.zebra_high_on);
    settings.setValue("Filter/zebra_high_level", g_AppSettings.zebra_high_level);
    settings.setValue("Filter/zebra_low_on", g_AppSettings.zebra_low_on);
    settings.setValue("Filter/zebra_low_level", g_AppSettings.zebra_low_level);
    settings.setValue("Screenshot/screenshot_format", g_AppSettings.screenshot_format);
    settings.setValue("Screenshot/screenshot_mode", g_AppSettings.screenshot_mode);
    settings.setValue("Screenshot/screenshot_tiff_compression", g_AppSettings.screenshot_tiff_compression);
    settings.setValue("Screenshot/screenshot_jpeg_quality", g_AppSettings.screenshot_jpeg_quality);
    settings.setValue("Screenshot/screenshot_jpeg_subsampling", g_AppSettings.screenshot_jpeg_subsampling);

    settings.setValue("Encodesetting/encode_path", encodeSettings.encode_path);
    settings.setValue("Encodesetting/container", encodeSettings.container);
    settings.setValue("Encodesetting/separate_files", encodeSettings.separate_files);
    settings.setValue("Encodesetting/audio_enable", encodeSettings.audio_enable);
    settings.setValue("Encodesetting/audio_codec", encodeSettings.audio_codec);
    settings.setValue("Encodesetting/audio_gain_db", encodeSettings.audio_gain_db);
    settings.setValue("Encodesetting/audio_bitrate", encodeSettings.audio_bitrate);
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
    settings.setValue("Encodesetting/bitrate_basis", encodeSettings.bitrate_basis);

    // ==========================
    // GPUInfo 保存
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
