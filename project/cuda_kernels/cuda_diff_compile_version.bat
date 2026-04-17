@echo off
setlocal EnableDelayedExpansion

rem ========================
rem *** NVCC PATH 設定 ***
rem ========================
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"

set "OUT_DIR=obj_out"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "TIME_FILE=%OUT_DIR%\last_build.txt"

rem ---- クリーンビルド判定 ----
if not exist "%TIME_FILE%" (
    echo No timestamp found. Clean build.
    del /q "%OUT_DIR%\*.obj" >nul 2>&1
    echo 0>"%TIME_FILE%"
)

set /p LAST_BUILD=<"%TIME_FILE%"
echo Last Build Ticks: !LAST_BUILD!

rem ---- PowerShellで更新された .cu のみビルド ----
for /f "delims=" %%F in ('powershell -NoProfile -Command ^
    "Get-ChildItem *.cu | Where-Object { $_.LastWriteTimeUtc.Ticks -gt [long]$env:LAST_BUILD } | ForEach-Object { $_.Name }"') do (
    
    echo   [BUILD] %%F

    "%NVCC%" -c "%%F" -o "%OUT_DIR%\%%~nF.obj" ^
        -gencode arch=compute_61,code=sm_61 ^
        -gencode arch=compute_75,code=sm_75 ^
        -gencode arch=compute_86,code=sm_86 ^
        -gencode arch=compute_89,code=sm_89 ^
        -gencode arch=compute_90,code=sm_90 ^
        -gencode arch=compute_120,code=sm_120 ^
        -Xcompiler="/MD /utf-8" ^
        --use_fast_math

    set "UPDATED=1"
)

rem ---- タイムスタンプ更新 ----
if defined UPDATED (
    for /f %%T in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().Ticks"') do set "NOW_TICKS=%%T"
    echo !NOW_TICKS!>"%TIME_FILE%"
    echo Updated build timestamp: !NOW_TICKS!
) else (
    echo No files were built.
)

pause
