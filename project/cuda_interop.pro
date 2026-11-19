message("PRO_FILE = $$PWD")
message("PRO_NAME = $$PRO_FILE_PWD")
message("OUT_PWD  = $$OUT_PWD")

QT += core gui opengl widgets
QT += widgets opengl
QT += opengl
QT += openglwidgets
LIBS += -lopengl32
win32: LIBS += -lpdh      # GPU の 3D 使用率（info_thread）
win32: LIBS += -ldxgi -ld3d11   # 表示経路の色深度（displayprobe.cpp。モニター出力・画面合成）

# ---- 外部ライブラリ（すべて 3rdparty/ に同梱）----
#   3rdparty/NVIDIAGPUComputingToolkit/CUDA/v13.0 : CUDA Toolkit（nvcc・cudart・NPP・NVML）
#   3rdparty/ffmpeg                               : FFmpeg（include / lib / bin）
#   3rdparty/gpuimg                               : gpuimg（画像保存。include / lib / bin）
#   3rdparty/nvapi                                : NVAPI（NVIDIA 公開の SDK、MIT）
#   実行時の DLL（各 bin/）は exe から見える場所（PATH か exe と同じフォルダ）に置くこと
THIRDPARTY_DIR = $$PWD/3rdparty
!exists($$THIRDPARTY_DIR/NVIDIAGPUComputingToolkit/CUDA/v13.0/include/cuda_runtime.h): \
    warning("CUDA Toolkit が見つかりません: $$THIRDPARTY_DIR/NVIDIAGPUComputingToolkit/CUDA/v13.0")
!exists($$THIRDPARTY_DIR/ffmpeg/include/libavcodec/avcodec.h): warning("FFmpeg が見つかりません: $$THIRDPARTY_DIR/ffmpeg")
!exists($$THIRDPARTY_DIR/gpuimg/include): warning("gpuimg が見つかりません: $$THIRDPARTY_DIR/gpuimg")

# nvCOMP（NVIDIA nvCOMP SDK 6.0）。マルチ GPU デコードの圧縮転送（試験的。src/videoprocess/xfer_codec.*）
#   実行時は bin/nvcomp64_6.dll（下の DEPLOY_DLLS で exe の横へコピー）。ライセンスは 3rdparty/nvcomp/LICENSE.txt
INCLUDEPATH += $$THIRDPARTY_DIR/nvcomp/include
win32: LIBS += -L$$THIRDPARTY_DIR/nvcomp/lib -lnvcomp

# NVAPI。displayprobe.cpp でドライバ設定 Deep color を読む
INCLUDEPATH += $$THIRDPARTY_DIR/nvapi
win32: LIBS += -L$$THIRDPARTY_DIR/nvapi/amd64 -lnvapi64


QT += widgets gui opengl
QT += multimedia
QT += concurrent
# Qt Quick 版 UI（--qml で起動）
QT += quick qml quickcontrols2

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += $$PWD/form
INCLUDEPATH += $$PWD/src
INCLUDEPATH += $$PWD/src/videoprocess
INCLUDEPATH += $$PWD/src/ui_control

SOURCES += \
    $$PWD/src/videoprocess/avidecode.cpp \
    $$PWD/src/videoprocess/cpudecode.cpp \
    $$PWD/src/videoprocess/nvgpudecode.cpp \
    $$PWD/src/videoprocess/decode_thread.cpp \
    $$PWD/src/videoprocess/audio_decode.cpp \
    $$PWD/src/videoprocess/save_encode.cpp \
    $$PWD/src/ui_control/audio_volume.cpp \
    $$PWD/src/ui_control/encode_setting.cpp \
    $$PWD/src/ui_control/jump_mode.cpp \
    $$PWD/src/ui_control/rangeslider.cpp \
    $$PWD/src/ui_control/video_speed.cpp \
    $$PWD/src/ui_control/save_screenshot.cpp \
    $$PWD/src/info/info_thread.cpp \
    $$PWD/src/info/logfile_control.cpp \
    $$PWD/src/imageprocess/cuda_imageprocess.cpp \
    $$PWD/src/main/fps_thread.cpp \
    $$PWD/src/main/glwidget.cpp \
    $$PWD/src/main/main.cpp \
    $$PWD/src/main/mainwindow.cpp \
    $$PWD/src/qmlui/videoengine.cpp \
    $$PWD/src/qmlui/videoitem.cpp \
    $$PWD/src/qmlui/displayprobe.cpp \
    $$PWD/src/qmlui/perfmeter.cpp \
    $$PWD/src/qmlui/decodesettings.cpp \
    $$PWD/src/videoprocess/xfer_codec.cpp \
    $$PWD/src/qmlui/playercontroller.cpp \
    $$PWD/src/qmlui/windowchrome.cpp \
    $$PWD/src/qmlui/encodecontroller.cpp \
    $$PWD/src/qmlui/screenshotsettings.cpp \
    $$PWD/src/qmlui/filterkernel.cpp \
    $$PWD/src/qmlui/tonecurve.cpp \
    $$PWD/src/qmlui/mediaprobe.cpp \

HEADERS += \
    $$PWD/src/__global__.h \
    $$PWD/src/videoprocess/avidecode.h \
    $$PWD/src/videoprocess/cpudecode.h \
    $$PWD/src/videoprocess/decode_thread.h \
    $$PWD/src/videoprocess/audio_decode.h \
    $$PWD/src/videoprocess/nvgpudecode.h \
    $$PWD/src/videoprocess/yuv_plane_ring.h \
    $$PWD/src/videoprocess/save_encode.h \
    $$PWD/src/ui_control/audio_volume.h \
    $$PWD/src/ui_control/encode_setting.h \
    $$PWD/src/ui_control/jump_mode.h \
    $$PWD/src/ui_control/rangeslider.h \
    $$PWD/src/ui_control/video_speed.h \
    $$PWD/src/ui_control/save_screenshot.h \
    $$PWD/src/info/info_thread.h \
    $$PWD/src/info/logfile_control.h \
    $$PWD/src/imageprocess/cuda_imageprocess.h \
    $$PWD/src/main/fps_thread.h \
    $$PWD/src/main/glwidget.h \
    $$PWD/src/main/mainwindow.h \
    $$PWD/src/qmlui/videoengine.h \
    $$PWD/src/qmlui/videoitem.h \
    $$PWD/src/qmlui/displayprobe.h \
    $$PWD/src/qmlui/perfmeter.h \
    $$PWD/src/qmlui/decodesettings.h \
    $$PWD/src/videoprocess/xfer_codec.h \
    $$PWD/src/videoprocess/decode_gpu_map.h \
    $$PWD/src/qmlui/playercontroller.h \
    $$PWD/src/qmlui/windowchrome.h \
    $$PWD/src/qmlui/encodecontroller.h \
    $$PWD/src/qmlui/screenshotsettings.h \
    $$PWD/src/qmlui/filterkernel.h \
    $$PWD/src/qmlui/tonecurve.h \
    $$PWD/src/qmlui/mediaprobe.h \
    src/main/__global__.h

FORMS += \
    $$PWD/form/audio_volume.ui \
    $$PWD/form/encode_setting.ui \
    $$PWD/form/jump_mode.ui \
    $$PWD/form/mainwindow.ui \
    $$PWD/form/video_speed.ui \
    $$PWD/form/save_screenshot.ui \

# ---- 表示の 10bit 検証（検証用ビルドだけ）----
#   qmake の引数に CONFIG+=display_verify を付けたときだけ組み込む（Qt Creator: プロジェクト → ビルド設定 →
#   qmake の「追加の引数」）。通常・リリースのビルドには入らない。Ctrl+Shift+D で表示の段数をログに出す
display_verify {
    DEFINES += CUDA_VIEWER_DISPLAY_VERIFY
    SOURCES += $$PWD/src/qmlui/displayverify.cpp
    HEADERS += $$PWD/src/qmlui/displayverify.h
}

# Qt Quick 版 UI の QML
RESOURCES += \
    $$PWD/qml.qrc

TRANSLATIONS += \
    cuda_interop_ja_JP.ts
CONFIG += lrelease
CONFIG += embed_translations

# #open cvリンク
# OPENCV_DIR = "$$PWD/OpenCV_withCUDA"

# INCLUDEPATH += $$OPENCV_DIR/include
# LIBS += -L$$OPENCV_DIR/x64/vc17/lib \
#         -lopencv_world4120  # ← ".lib" は不要！

#CUDAリンク
CUDA_DIR = "$$THIRDPARTY_DIR/NVIDIAGPUComputingToolkit/CUDA/v13.0"

INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib/x64 \
        -lcudart \
        -lnppc \
        -lnppial \
        -lnppicc \
        -lnppidei \
        -lnppif \
        -lnppig \
        -lnppim \
        -lnppist \
        -lnppisu \
        -lnvml        # GPU の使用率など（info_thread）。nvml.lib は CUDA Toolkit の lib/x64 に入っている

# ---- CUDA カーネル（cuda_kernels/*.cu）を qmake のビルドで nvcc に通す ----
#   以前は cuda_kernels/*.bat で obj_out/*.obj を作ってリンクしていた（bat は不要になった）。
#   ・コンパイルするのは下の CUDA_SOURCES に書いたファイルだけ（増やしたらここに足して qmake）
#     ファイル名の 8bit / 10bit は処理経路（GL テクスチャの形式）を表す
#       8bit 経路  = RGBA8（8bit 素材。Widgets 版も常にこちら）
#       10bit 経路 = RGBA16（10bit 以上の素材。QML 版で自動で切り替え）
#       decode_8bit.cu     : デコード結果（NV12 / YUV / RGB。入力は 8bit / 10bit）→ RGBA8 テクスチャ
#       encode_8bit.cu     : RGBA8 の FBO → NV12（エンコード入力）
#       histogram_8bit.cu  : RGBA8 の FBO からヒストグラム
#       decode_10bit.cu    : P010（NVDEC）/ 10bit プレーン（CPU デコード）→ RGBA16 テクスチャ
#       encode_10bit.cu    : RGBA16 の FBO → NV12（エンコード入力）/ RGBA8（画像保存）
#       histogram_10bit.cu : RGBA16 の FBO からヒストグラム（1024 段。集計先の構造体は histogram_10bit.h）
#       tiles_10bit.h      : 10bit 経路のカーネルへ渡すタイルの構造体（cuda_imageprocess.cpp と共用）
#       decode_rows.cuh    : NVDEC のタイル → RGBA の行ごとの共通部（decode_8bit / decode_10bit。デコードの本線）
#       encode_rows.cuh    : RGBA → NV12 / P010 のタイルの行ごとの共通部（encode_8bit / encode_10bit。エンコードの本線）
#       xfer_compress.cu   : マルチ GPU の圧縮転送（差分・詰め直し・圧縮チャンクの受け渡し。xfer_codec.cpp から）
#     起動口は src/imageprocess/cuda_imageprocess.cpp
#   ・.cu を変更しただけなら通常のビルドで、変わったものだけ再コンパイルされる
#   ・cuda_kernels/ の .cuh / .h を変えると、全カーネルを再コンパイルする
#   ・nvcc はプロジェクト同梱の CUDA_DIR/bin を優先（無ければ PATH の nvcc）
#   ・cl.exe は Qt Creator の MSVC キットの環境から見つかる
#   ・対象 GPU: Turing（sm_75）以降。CUDA 13 は Pascal（sm_61）以前を扱えない
CUDA_SOURCES = \
    $$PWD/cuda_kernels/decode_8bit.cu \
    $$PWD/cuda_kernels/encode_8bit.cu \
    $$PWD/cuda_kernels/histogram_8bit.cu \
    $$PWD/cuda_kernels/decode_10bit.cu \
    $$PWD/cuda_kernels/encode_10bit.cu \
    $$PWD/cuda_kernels/histogram_10bit.cu \
    $$PWD/cuda_kernels/xfer_compress.cu
CUDA_HEADERS = $$files($$PWD/cuda_kernels/*.cuh) $$files($$PWD/cuda_kernels/*.h)
CUDA_ARCHS   = 75 86 89 90 120

NVCC = $$CUDA_DIR/bin/nvcc.exe
!exists($$NVCC): NVCC = nvcc
NVCC = $$shell_path($$NVCC)

for(arch, CUDA_ARCHS): CUDA_GENCODE += -gencode arch=compute_$${arch},code=sm_$${arch}
# ランタイムライブラリは本体のビルド（Debug は /MDd、Release は /MD）に合わせる。
#   出力先も分ける（同じビルドフォルダで Debug / Release を切り替えても古い obj を使わない）
CONFIG(debug, debug|release) {
    CUDA_CRT     = /MDd
    CUDA_OBJ_DIR = $$OUT_PWD/cuda_obj/debug
} else {
    CUDA_CRT     = /MD
    CUDA_OBJ_DIR = $$OUT_PWD/cuda_obj/release
}
!exists($$CUDA_OBJ_DIR): !mkpath($$CUDA_OBJ_DIR): error("CUDA の出力フォルダを作れません: $$CUDA_OBJ_DIR")

cuda.input              = CUDA_SOURCES
cuda.output             = $$CUDA_OBJ_DIR/${QMAKE_FILE_BASE}_cu.obj
cuda.commands           = $$NVCC -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} $$CUDA_GENCODE \
                          -Xcompiler $${CUDA_CRT},/utf-8 --use_fast_math
cuda.depends            = $$CUDA_HEADERS
cuda.dependency_type    = TYPE_C
cuda.variable_out       = OBJECTS
cuda.name               = nvcc ${QMAKE_FILE_IN}
QMAKE_EXTRA_COMPILERS  += cuda

#ffmpeg
ffmpeg_DIR = "$$THIRDPARTY_DIR/ffmpeg"

INCLUDEPATH += $$ffmpeg_DIR/include
LIBS += -L$$ffmpeg_DIR/lib

LIBS += -lavformat
LIBS += -lavcodec
LIBS += -lavfilter        # もし使うなら
LIBS += -lswresample
LIBS += -lswscale
LIBS += -lavutil

#gpuimg
gpuimg_DIR = "$$THIRDPARTY_DIR/gpuimg"

INCLUDEPATH += $$gpuimg_DIR/include
LIBS += -L$$gpuimg_DIR/lib

LIBS += -lgpuimg


# ---- 実行に必要な DLL をビルド後に exe と同じフォルダへコピー（Windows）----
#   3rdparty/ の各 bin から、このアプリが読み込む DLL だけをコピーする（xcopy /D = 新しいときだけ上書き）。
#   ・リンク（exe の作り直し）のときに動く。exe が最新で DLL だけ無い場合は「リビルド」すること
#   ・nvml.dll / nvapi64.dll / nvcuda.dll は NVIDIA のドライバに入っているのでコピーしない
#   ・DLL を足したら下の DEPLOY_DLLS に足す
win32 {
    # exe の出力先（debug_and_release なら debug / release のサブフォルダ）
    EXE_DIR = $$OUT_PWD
    !isEmpty(DESTDIR) {
        EXE_DIR = $$DESTDIR
    } else: debug_and_release {
        CONFIG(debug, debug|release): EXE_DIR = $$OUT_PWD/debug
        else:                         EXE_DIR = $$OUT_PWD/release
    }

    # 場所は THIRDPARTY_DIR から直接組み立てる（CUDA_DIR などは "" 付きの値なので、そのまま繋ぐと引用符が混ざる）
    CUDA_BIN_DIR   = $$THIRDPARTY_DIR/NVIDIAGPUComputingToolkit/CUDA/v13.0/bin/x64
    FFMPEG_BIN_DIR = $$THIRDPARTY_DIR/ffmpeg/bin
    GPUIMG_BIN_DIR = $$THIRDPARTY_DIR/gpuimg/bin
    DEPLOY_DLLS = \
        $$CUDA_BIN_DIR/cudart64_13.dll \
        $$CUDA_BIN_DIR/nppc64_13.dll \
        $$CUDA_BIN_DIR/nppial64_13.dll \
        $$CUDA_BIN_DIR/nppicc64_13.dll \
        $$CUDA_BIN_DIR/nppidei64_13.dll \
        $$CUDA_BIN_DIR/nppif64_13.dll \
        $$CUDA_BIN_DIR/nppig64_13.dll \
        $$CUDA_BIN_DIR/nppim64_13.dll \
        $$CUDA_BIN_DIR/nppist64_13.dll \
        $$CUDA_BIN_DIR/nppisu64_13.dll \
        $$FFMPEG_BIN_DIR/avcodec-62.dll \
        $$FFMPEG_BIN_DIR/avformat-62.dll \
        $$FFMPEG_BIN_DIR/avfilter-11.dll \
        $$FFMPEG_BIN_DIR/avdevice-62.dll \
        $$FFMPEG_BIN_DIR/avutil-60.dll \
        $$FFMPEG_BIN_DIR/swresample-6.dll \
        $$FFMPEG_BIN_DIR/swscale-9.dll \
        $$GPUIMG_BIN_DIR/gpuimg.dll \
        $$THIRDPARTY_DIR/nvcomp/bin/nvcomp64_6.dll

    for(dll, DEPLOY_DLLS) {
        !exists($$dll): warning("コピーする DLL が見つかりません: $$dll")
        # コピー先はフォルダ（リンク後なので必ずある）。末尾に \ を付けない（"…\" が引用符の扱いを壊す）
        QMAKE_POST_LINK += xcopy /D /Y /Q $$shell_quote($$shell_path($$dll)) $$shell_quote($$shell_path($$EXE_DIR)) >nul $$escape_expand(\n\t)
    }
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
