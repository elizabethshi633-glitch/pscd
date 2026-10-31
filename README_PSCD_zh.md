# PSCD-GS 复现指南

本仓库实现 **PSCD-GS（Role-Aware Perception-Structure Collaborative
Densification）** 的角色感知增密训练流程。本文档说明如何准备运行环境和
数据、训练与评估模型，并以可重复的方式运行 PSCD。

> 本文档以当前代码为准。复现 PSCD 时必须传入 `--pscd`。

## 1. 仓库功能

给定一个带相机位姿的多视角数据集，训练程序以 COLMAP 稀疏点云初始化一组可
学习的三维各向异性高斯。训练同时优化位置、尺度、旋转、不透明度和球谐颜色，
再通过 CUDA 高斯光栅器渲染新视角。

PSCD 在每个预定增密事件中额外执行以下步骤：

1. 为训练图像计算冻结 VGG-16 特征响应（或 Sobel 后备特征）与二维方向张量；
2. 从当前高斯模型渲染深度，结合图像边缘形成结构响应；
3. 用相邻相机的重投影深度/结构一致性过滤不可靠区域；
4. 用高斯的 alpha 合成贡献，将感知证据和结构证据汇总到每个高斯；
5. 生成感知主导、结构主导、共同主导等连续角色，以有限预算选择 clone 或
   split 候选；对 split 候选沿可信三维方向放置子高斯，随后执行不透明度和尺度
   剪枝。

训练结果以 PLY 格式保存，可由 `render.py` 和 SIBR 查看器读取。

## 2. 运行环境

### 必要硬件

- NVIDIA CUDA GPU，计算能力建议不低于 7.0；
- 论文质量设置建议约 24 GB 显存。较小显存可先用低分辨率、小数据集或
  `--data_device cpu` 验证流程；
- 足够的磁盘空间保存 Conda 环境、编译产物、图像和输出模型。

CPU 或没有可用 CUDA 驱动的机器无法训练、渲染或计算指标，因为主代码和两个
自定义扩展均直接使用 CUDA。

### 软件版本

| 组件 | 当前仓库的基准版本/要求 |
| --- | --- |
| 操作系统 | Windows 10/11 或 Ubuntu 22.04 |
| Python | 3.7.13 |
| PyTorch | 1.12.1 |
| torchvision | 0.13.1 |
| Conda / Mamba | 推荐，用于创建隔离环境 |
| CUDA 运行时 | `environment.yml` 固定为 11.6 |
| CUDA Toolkit | 11.x；仓库原说明推荐 11.8，并指出 11.6 在部分 Windows 环境有编译问题 |
| C++ 编译器 | Windows 使用 Visual Studio 2019/Build Tools 的 MSVC；Linux 使用兼容的 GCC |

`cudatoolkit` 是 Conda 中的运行时；编译 `diff-gaussian-rasterization` 和
`simple-knn` 仍需要本机 CUDA Toolkit、匹配的编译器和 PyTorch。不要把彼此主
版本不兼容的 PyTorch、CUDA Toolkit 与 CUDA 运行时混用。

### Windows 安装

在 **x64 Native Tools Command Prompt for Visual Studio** 或已配置 MSVC 的终端中，
于仓库根目录执行：

```powershell
Set-Location C:\path\to\gaussian-splatting
$env:DISTUTILS_USE_SDK = "1"
conda env create --file environment.yml
conda activate gaussian_splatting
```

`environment.yml` 已列出两个 CUDA 扩展的安装项，首次创建环境时会编译它们。
如环境创建完成但扩展导入失败，可在已激活的环境中重装：

```powershell
pip install --no-build-isolation .\submodules\diff-gaussian-rasterization
pip install --no-build-isolation .\submodules\simple-knn
```

若压缩包来自不含子模块的 Git 克隆，先补齐子模块：

```powershell
git submodule update --init --recursive
```

### Linux 安装

确保 `nvcc`、CUDA 驱动和 C++ 编译器可用后：

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

### 安装验证

在激活 `gaussian_splatting` 环境后运行：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print('CUDA extensions: OK')"
python train.py --help
```

第一条命令中的 `torch.cuda.is_available()` 必须为 `True`。第二条命令必须没有
报错；否则不要开始训练。Windows 上已有的 Linux `.so` 或不同 Python 版本编译
的产物不可复用，必须在当前环境重新编译。

## 3. 数据集

### 支持的数据格式

#### COLMAP 数据集

训练目录至少应具有如下布局：

```text
<scene>/
├── images/                         # 输入图像
└── sparse/0/
    ├── cameras.bin 或 cameras.txt
    ├── images.bin  或 images.txt
    └── points3D.bin 或 points3D.txt
```

首次读取时，程序会自动将 `points3D.bin`/`points3D.txt` 转成
`sparse/0/points3D.ply`。使用 `--eval` 时，按图像名称排序后每 8 张留 1 张作为
测试集；其余图像用于训练。

#### NeRF Synthetic / Blender 数据集

目录包含 `transforms_train.json` 与 `transforms_test.json`，并由 JSON 引用 PNG
图像。透明区域可配合 `--white_background` 使用。该格式没有 COLMAP 点云时，代码
会生成随机初始点。

### 仓库自带示例

根目录的 `pack.zip` 包含一个名为 `test/` 的 COLMAP 示例数据集。为避免覆盖现有
内容，可解压到新目录：

```powershell
New-Item -ItemType Directory -Path .\data -ErrorAction Stop
Expand-Archive -LiteralPath .\pack.zip -DestinationPath .\data
```

之后的示例数据路径为 `.\data\test`。该数据适合检查流程能否跑通，不等同于论文
评测数据集。

### 自己的图片或视频

将图像放入 `<scene>/input/`，安装 COLMAP 后执行：

```powershell
python convert.py -s <scene>
```

该命令执行 SfM 与去畸变，并生成训练所需的 `images/` 和 `sparse/0/`。如需要
预生成不同分辨率的图像，增加 `--resize`；如已完成匹配，增加 `--skip_matching`。

### 论文级评测数据

建议使用 MipNeRF360、Tanks & Temples 和 Deep Blending 场景完成大规模评测。
这些数据集不随本仓库分发，应从各自的官方来源获取并遵守其许可。`full_eval.py`
中列出的场景为：

- MipNeRF360：`bicycle`、`flowers`、`garden`、`stump`、`treehill`、`room`、
  `counter`、`kitchen`、`bonsai`；
- Tanks & Temples：`truck`、`train`；
- Deep Blending：`drjohnson`、`playroom`。

## 4. 训练、渲染和评估

### 复现 PSCD-GS

使用 VGG-16 感知分支的主复现实验：

```powershell
python train.py -s .\data\test --eval -m .\output\pscd_vgg16 --iterations 30000 --pscd --pscd_perceptual_backbone vgg16 --checkpoint_iterations 7000 30000
```

首次使用 `vgg16` 时，torchvision 必须能取得 `VGG16_Weights.IMAGENET1K_V1` 并将其
缓存到本机；离线机器请预先缓存该权重。若只需验证 PSCD 管线可执行，可改用：

```powershell
python train.py -s .\data\test --eval -m .\output\pscd_sobel --iterations 30000 --pscd --pscd_perceptual_backbone sobel
```

`sobel` 保留角色感知增密的其余流程，但它是工程后备方案，**不是**与 VGG-16
论文设置可直接比较的复现结果。

训练完成后，输出目录包含：

```text
<model_path>/
├── cfg_args                         # 实际命令行配置
├── cameras.json                     # 相机记录
├── input.ply                        # 初始点云副本
├── point_cloud/iteration_<N>/point_cloud.ply
└── chkpnt<N>.pth                    # 仅在指定 --checkpoint_iterations 时生成
```

从模型目录渲染并计算测试集指标：

```powershell
python render.py -m .\output\pscd_vgg16 --iteration 30000 --skip_train
python metrics.py -m .\output\pscd_vgg16
```

结果会写入 `<model_path>/results.json` 与 `<model_path>/per_view.json`，包括
SSIM、PSNR 和 LPIPS。

## 5. PSCD 默认配置与复现控制

除 `--pscd` 外，其余 PSCD 参数均已有代码默认值。关键默认项如下：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--pscd_perceptual_backbone` | `vgg16` | 感知证据提取器；`sobel` 仅为离线后备 |
| `--pscd_neighbors` | 4 | 每个相机使用的邻居视角数 |
| `--pscd_q_low` / `--pscd_q_high` | 0.05 / 0.95 | 证据图的稳健分位数归一化范围 |
| `--pscd_edge_weight` / `--pscd_depth_weight` | 3/7 / 4/7 | 结构证据中图像边缘与深度边缘的权重 |
| `--pscd_visibility_views` | 3 | 证据汇总的可见视角目标数 |
| `--pscd_priority_threshold` | 0.6 | 进入增密候选集的最低优先级 |
| `--pscd_growth_rate` | 0.05 | 每次增密最多新增当前高斯数的比例 |
| `--pscd_max_candidates` | 50,000 | 每次增密的候选上限 |
| `--pscd_max_gaussians` | 5,000,000 | 高斯总数安全上限 |
| `--pscd_split_threshold` | 0.5 | 选择 split 而不是 clone 的阈值 |
| `--pscd_split_displacement` | 0.5 | split 子高斯沿估计方向的位移倍率 |

完整参数及默认值可用下列命令查看：

```powershell
python train.py --help
```

为保证实验可比较，请固定以下条件：

1. 所有待比较的 PSCD 实验使用同一份解压后的数据、相同 `--eval` 划分、图像目录和分辨率；
2. 固定训练为 30,000 次，并记录全部优化参数；
3. 保留输出目录中的 `cfg_args`、PyTorch/CUDA 版本和 GPU 型号；
4. 代码在启动时固定 Python、NumPy 与 PyTorch 的随机种子为 0，但 CUDA 算子仍可能
   在不同显卡、驱动或库版本下出现细微非确定性；
5. 报告结果时明确感知分支是 `vgg16` 还是 `sobel`，不要将后者作为论文 VGG 设置的
   等价结果。

## 6. 批量评测

`full_eval.py` 可对三套公开基准批量进行训练、渲染和指标评估：

```powershell
python full_eval.py -m360 <mipnerf360目录> -tat <tanksandtemples目录> -db <deepblending目录> --output_path .\eval
```

该脚本当前没有 `--pscd` 参数，不能直接用于 PSCD 批量复现。请为每个场景显式
调用 `train.py ... --pscd`，或在确认实验方案后为 `full_eval.py` 的 `common_args`
补充 PSCD 参数。

## 7. 常见问题

- **`No module named torch`**：未激活 Conda 环境，或环境创建失败。
- **找不到 `nvcc`、`cl.exe` 或编译扩展失败**：安装/配置匹配的 CUDA Toolkit 和
  Visual Studio Build Tools，并在配置 MSVC 的终端中重建扩展。
- **`diff_gaussian_rasterization` 或 `simple_knn` 导入失败**：删除与当前系统、
  Python 或 CUDA 不匹配的旧构建产物后，按“Windows 安装”节重新执行两条 `pip install`。
- **VGG-16 权重错误**：联网预缓存 torchvision 的 ImageNet VGG-16 权重，或仅为流程
  验证使用 `--pscd_perceptual_backbone sobel`。
- **显存不足**：降低图像分辨率（`-r 2`、`-r 4` 等）、使用 `--data_device cpu`，或先
  使用较小场景；PSCD 在增密时会进行额外深度和贡献统计，因而显存与时间开销较高。
- **数据集无法识别**：确认根目录含 `sparse/`（COLMAP）或 `transforms_train.json`
  （Blender），并检查 `images/` 路径与 `--images` 参数一致。

## 8. 重要限制

- 该仓库的许可为非商业、研究和评估用途，使用前请阅读 `LICENSE.md`；
- 当前仓库没有提供 PSCD 论文的预训练权重、完整实验日志或自动化批量 PSCD 脚本；
- 仅凭代码默认值不能保证与某篇论文数字完全一致。相机标定、数据版本、预处理、
  显卡、驱动、CUDA、PyTorch 与 VGG 权重版本都应记录在实验报告中。
