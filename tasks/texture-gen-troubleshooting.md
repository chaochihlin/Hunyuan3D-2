# Hunyuan3D-2 Texture Generation 問題排查與修正紀錄

## 環境

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Ti (16GB VRAM, sm_120, Compute 12.0) |
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 / 13.2（雙版本並存） |
| MSVC | VS 2026 (v18, MSVC 14.50) + VS 2022 BuildTools (MSVC 14.44) |

## 原始問題

Texture generation 執行超過 30 分鐘未完成，嚴重不合理。

---

## 問題 1：VRAM 爆滿導致極端緩慢

### 原因

`minimal_demo.py` 在一開始就同時載入 shapegen 和 texgen 兩條 pipeline：

```python
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
```

Texgen 內部又載入兩個 diffusion model（delight + multiview），三組模型同時佔用 VRAM 遠超 16GB，導致大量 CPU↔GPU swap，速度暴跌。

### 修正

改為先完成 shape generation，釋放 VRAM 後再載入 texgen：

```python
# Shape generation
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
mesh = pipeline_shapegen(image=image)[0]

# 釋放 shapegen VRAM
del pipeline_shapegen
torch.cuda.empty_cache()

# 再載入 texture generation
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
mesh = pipeline_texgen(mesh, image=image)
```

**修改檔案**：`minimal_demo.py`

---

## 問題 2：custom_rasterizer CUDA kernel 不支援 sm_120

### 現象

修完 VRAM 問題後跑起來，在 texture baking 階段 crash：

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

### 原因

`custom_rasterizer_kernel.pyd` 編譯時沒有包含 sm_120（RTX 5070 Ti 的 compute capability 12.0），需要重新編譯。

### 重新編譯遇到的連環障礙

#### 障礙 2a：CUDA 版本不匹配

直接跑 `setup.py` 會被 PyTorch 擋掉：

```
RuntimeError: The detected CUDA version (13.2) mismatches the version that was used to compile PyTorch (12.8)
```

**解法**：指定 `CUDA_HOME` 使用 CUDA 12.8 而非系統預設的 13.2。

#### 障礙 2b：CUDA 12.8 nvcc 不支援 VS 2026

```
fatal error C1189: #error: unsupported Microsoft Visual Studio version!
Only the versions between 2017 and 2022 (inclusive) are supported!
```

**解法**：在 `setup.py` 加上 `--allow-unsupported-compiler` flag。

#### 障礙 2c：CUDA 13.2 headers 與 PyTorch 12.8 headers 衝突（死路）

曾嘗試改用 CUDA 13.2 編譯（繞過 VS 版本問題），但 CUDA 13.2 的 CCCL/thrust headers 和 PyTorch 附帶的 CUDA 12.8 headers 互相衝突，產生大量 `expected a "{"` 錯誤。**此路不通**。

#### 障礙 2d：VS 2026 MSVC 與 PyTorch headers 的 `std` 歧義

用 CUDA 12.8 + `--allow-unsupported-compiler` + VS 2026 MSVC (v14.50) 編譯時：

```
torch/csrc/dynamo/compiled_autograd.h(1143): error C2872: 'std': ambiguous symbol
```

**原因**：`.cu` 檔案透過 `#include <torch/extension.h>` 引入了完整的 PyTorch header chain，其中 `compiled_autograd.h` 在 VS 2026 下觸發 `std` 命名空間歧義。

**解法**：修改 `rasterizer.h`，將 `#include <torch/extension.h>` 替換為更輕量的 `#include <torch/types.h>`（CUDA kernel 本身不需要完整的 torch extension headers），然後在 `rasterizer.cpp`（負責 pybind11 binding）中補回 `#include <torch/extension.h>`。

### 最終成功的編譯配置

| 組件 | 版本 |
|------|------|
| nvcc | CUDA 12.8 |
| Host compiler (cl.exe) | VS 2026 MSVC 14.50 |
| Target arch | sm_120 (TORCH_CUDA_ARCH_LIST=12.0) |
| 特殊 flag | `--allow-unsupported-compiler` |

**修改檔案**：
- `hy3dgen/texgen/custom_rasterizer/setup.py` — 加上 `--allow-unsupported-compiler`
- `hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.h` — `torch/extension.h` → `torch/types.h`
- `hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp` — 補回 `#include <torch/extension.h>`

**編譯腳本**：`rebuild_rasterizer.bat`

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=12.0
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%
cd /d C:\Users\User\work\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer
rmdir /s /q build 2>nul
python setup.py build_ext --inplace
```

安裝：`install_rasterizer.bat`（同樣環境變數，改跑 `setup.py install`）

---

## 修正結果

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| Texture generation 時間 | 30+ 分鐘（未完成） | ~4.5 分鐘 |
| 輸出檔案 | 無 | demo.glb (22.6MB) |
| 貼圖 | 無 | 2048×2048 PBR texture |
| 頂點數 | - | 509,460 |
| 面數 | - | 678,648 |
