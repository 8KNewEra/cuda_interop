@echo off
setlocal

set OUT_DIR=obj_out

REM --- 出力フォルダが無ければ作成 ---
if not exist %OUT_DIR% mkdir %OUT_DIR%

REM --- 古い OBJ を削除 ---
echo Cleaning old OBJ files in %OUT_DIR% ...
del /q "%OUT_DIR%\*.obj" 2>nul

echo.

for %%f in (*.cu) do (
    echo Compiling %%f ...

    nvcc -c "%%f" -o "%OUT_DIR%\%%~nf.obj" ^
         -gencode arch=compute_75,code=sm_75 ^
         -gencode arch=compute_86,code=sm_86 ^
         -gencode arch=compute_89,code=sm_89 ^
         -gencode arch=compute_90,code=sm_90 ^
	 -gencode arch=compute_120,code=sm_120 ^
         -Xcompiler="/MD /utf-8" ^
         --use_fast_math
)

echo.
echo === Done! OBJ generated in %OUT_DIR% ===
pause
