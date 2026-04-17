@echo off
setlocal

set OUT_DIR=obj_out
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"

if not exist %OUT_DIR% mkdir %OUT_DIR%

echo Cleaning old OBJ files in %OUT_DIR% ...
del /q "%OUT_DIR%\*.obj" 2>nul
echo.

for %%f in (*.cu) do (
    echo Compiling %%f using CUDA 12.8...
    
    %NVCC% -c "%%f" -o "%OUT_DIR%\%%~nf.obj" ^
        -gencode arch=compute_61,code=sm_61 ^
        -gencode arch=compute_75,code=sm_75 ^
        -gencode arch=compute_86,code=sm_86 ^
        -gencode arch=compute_89,code=sm_89 ^
        -gencode arch=compute_90,code=sm_90 ^
	-gencode arch=compute_120,code=sm_120 ^
        -Xcompiler="/MD /utf-8" ^
        --use_fast_math
)

echo.
echo === Done! (CUDA 12.8 NVCC used) OBJ generated in %OUT_DIR% ===
pause

