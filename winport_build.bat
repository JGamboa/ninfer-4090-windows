@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d E:\LLM\ninfer-4090-winport
cmake --build build -j 1>build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
