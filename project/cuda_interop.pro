message("PRO_FILE = $$PWD")
message("PRO_NAME = $$PRO_FILE_PWD")
message("OUT_PWD  = $$OUT_PWD")

QT += core gui opengl widgets
QT += widgets opengl
QT += opengl
QT += openglwidgets
LIBS += -lopengl32


QT += widgets gui opengl
QT += multimedia
QT += concurrent

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
    $$PWD/src/videoprocess/save_encode.cpp \
    $$PWD/src/ui_control/audio_volume.cpp \
    $$PWD/src/ui_control/encode_setting.cpp \
    $$PWD/src/ui_control/jump_mode.cpp \
    $$PWD/src/ui_control/rangeslider.cpp \
    $$PWD/src/ui_control/video_speed.cpp \
    $$PWD/src/info/info_thread.cpp \
    $$PWD/src/info/logfile_control.cpp \
    $$PWD/src/imageprocess/cuda_imageprocess.cpp \
    $$PWD/src/main/fps_thread.cpp \
    $$PWD/src/main/glwidget.cpp \
    $$PWD/src/main/main.cpp \
    $$PWD/src/main/mainwindow.cpp

HEADERS += \
    $$PWD/src/__global__.h \
    $$PWD/src/videoprocess/avidecode.h \
    $$PWD/src/videoprocess/cpudecode.h \
    $$PWD/src/videoprocess/decode_thread.h \
    $$PWD/src/videoprocess/nvgpudecode.h \
    $$PWD/src/videoprocess/save_encode.h \
    $$PWD/src/ui_control/audio_volume.h \
    $$PWD/src/ui_control/encode_setting.h \
    $$PWD/src/ui_control/jump_mode.h \
    $$PWD/src/ui_control/rangeslider.h \
    $$PWD/src/ui_control/video_speed.h \
    $$PWD/src/info/info_thread.h \
    $$PWD/src/info/logfile_control.h \
    $$PWD/src/imageprocess/cuda_imageprocess.h \
    $$PWD/src/main/fps_thread.h \
    $$PWD/src/main/glwidget.h \
    $$PWD/src/main/mainwindow.h \
    src/main/__global__.h

FORMS += \
    $$PWD/form/audio_volume.ui \
    $$PWD/form/encode_setting.ui \
    $$PWD/form/jump_mode.ui \
    $$PWD/form/mainwindow.ui \
    $$PWD/form/video_speed.ui

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
CUDA_DIR = "$$PWD/NVIDIAGPUComputingToolkit/CUDA/v13.0"

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
        -lnppisu

LIBS += -L"$$PWD/NVIDIA Corporation/NVSMI" -lnvml

#CUDAの.objファイルを列挙してリンク
CUDA_OBJ_OUT_DIR = $$PWD/cuda_kernels/obj_out/
CUDA_OBJ_FILES = $$files($$CUDA_OBJ_OUT_DIR/*.obj)
for(obj, CUDA_OBJ_FILES) {
    LIBS += $$obj
}

#ffmepg
ffmpeg_DIR = "$$PWD/ffmpeg"

INCLUDEPATH += $$ffmpeg_DIR/include
LIBS += -L$$ffmpeg_DIR/lib

LIBS += -lavformat
LIBS += -lavcodec
LIBS += -lavfilter        # もし使うなら
LIBS += -lswresample
LIBS += -lswscale
LIBS += -lavutil


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
