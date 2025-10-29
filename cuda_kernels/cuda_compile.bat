@echo off
setlocal

set OUT_DIR=obj_out
if not exist %OUT_DIR% mkdir %OUT_DIR%

for %%f in (*.cu) do (
    echo Compiling %%f ...
    nvcc -c "%%f" -o "%OUT_DIR%\%%~nf.obj" ^
         -gencode arch=compute_75,code=sm_75 ^
         -gencode arch=compute_86,code=sm_86 ^
         -gencode arch=compute_89,code=sm_89 ^
         -gencode arch=compute_90,code=sm_90 ^
         -Xcompiler "/MD" ^
         --use_fast_math
)

echo.
echo === Done! OBJ generated in %OUT_DIR% ===
pause


