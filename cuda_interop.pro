QT += core gui opengl widgets
QT += widgets opengl
QT += opengl
QT += openglwidgets
LIBS += -lopengl32


QT += widgets gui opengl
QT += concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    cuda_imageprocess.cpp \
    decode_thread.cpp \
    fps_thread.cpp \
    glwidget.cpp \
    info_thread.cpp \
    main.cpp \
    mainwindow.cpp \
    save_encode.cpp

HEADERS += \
    cuda_imageprocess.h \
    decode_thread.h \
    fps_thread.h \
    glwidget.h \
    info_thread.h \
    mainwindow.h \
    save_encode.h

FORMS += \
    mainwindow.ui

TRANSLATIONS += \
    cuda_interop_ja_JP.ts
CONFIG += lrelease
CONFIG += embed_translations

#open cvリンク
OPENCV_DIR = "$$PWD/OpenCV_withCUDA"

INCLUDEPATH += $$OPENCV_DIR/include
LIBS += -L$$OPENCV_DIR/x64/vc17/lib \
        -lopencv_world4120  # ← ".lib" は不要！

#CUDAリンク
CUDA_DIR = "$$PWD/NVIDIAGPUComputingToolkit/CUDA/v12.8"

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

LIBS += -lavcodec -lavformat -lavutil -lswscale -lswresample

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
