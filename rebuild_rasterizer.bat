@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=12.0
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%
cd /d C:\Users\User\work\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer
rmdir /s /q build 2>nul
C:\Users\User\work\Hunyuan3D-2\venv\Scripts\python.exe setup.py build_ext --inplace 2>&1
