# PSCD-GS

PSCD-GS（Role-Aware Perception-Structure Collaborative Densification）实现角色感知协同增密：训练会汇总感知特征、图像/深度结构响应与跨视角一致性，在受预算约束的增密阶段选择 clone 或 split 候选。

## 运行环境

| 项目 | 建议配置 |
| --- | --- |
| 系统 | Windows 10/11 或 Ubuntu 22.04 |
| GPU | NVIDIA CUDA GPU，计算能力建议不低于 7.0；正式实验建议约 24 GB 显存 |
| Python | 3.7.13 |
| PyTorch / torchvision | 1.12.1 / 0.13.1 |
| CUDA | Conda 运行时 11.6；本机 CUDA Toolkit 使用兼容的 11.x 版本 |
| 编译器 | Windows：Visual Studio 2019 Build Tools/MSVC；Linux：兼容的 GCC |

CPU 或未正确安装 CUDA 驱动的机器不能训练、渲染或计算指标。CUDA 扩展 diff-gaussian-rasterization 与 simple-knn 会在创建环境时编译。

## 安装与验证

~~~powershell
git clone --recursive https://github.com/elizabethshi633-glitch/pscd.git
Set-Location .\pscd
conda env create --file environment.yml
conda activate gaussian_splatting
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print('CUDA extensions: OK')"
python train.py --help
~~~

Windows 下如果扩展未成功构建，请在激活环境并配置 MSVC 后执行：

~~~powershell
$env:DISTUTILS_USE_SDK = "1"
pip install --no-build-isolation .\submodules\diff-gaussian-rasterization
pip install --no-build-isolation .\submodules\simple-knn
~~~

其中 torch.cuda.is_available() 必须输出 True，两个扩展导入也不能报错。

## 数据集

COLMAP 格式训练目录：

~~~text
<scene>/
├── images/
└── sparse/0/
    ├── cameras.bin 或 cameras.txt
    ├── images.bin  或 images.txt
    └── points3D.bin 或 points3D.txt
~~~

使用 --eval 时，按图像名排序后每 8 张保留 1 张为测试图像。程序首次读取时会生成 sparse/0/points3D.ply。

Blender 格式应包含 transforms_train.json、transforms_test.json 与 JSON 引用的 PNG 图像；透明背景可使用 --white_background。

从图片或视频构建 COLMAP 数据：

~~~powershell
python convert.py -s <scene>
~~~

该命令会生成 images/ 与 sparse/0/；可按需加入 --resize 或 --skip_matching。

本地发布包可能附带 pack.zip 示例数据。该大文件不会被 GitHub 仓库追踪；如已获得它，可解压到 data/ 后用其中的 test/ 场景验证流程。

## 复现 PSCD-GS

主实验（VGG-16 感知分支）：

~~~powershell
python train.py -s .\data\test --eval -m .\output\pscd_vgg16 --iterations 30000 --pscd --pscd_perceptual_backbone vgg16 --checkpoint_iterations 7000 30000
~~~

--pscd 是启用角色感知增密的必要参数。首次使用 vgg16 时，torchvision 需要下载并缓存 VGG16_Weights.IMAGENET1K_V1；离线环境请提前准备该权重。

仅验证完整训练管线时可使用 Sobel 后备特征：

~~~powershell
python train.py -s .\data\test --eval -m .\output\pscd_sobel --iterations 30000 --pscd --pscd_perceptual_backbone sobel
~~~

## 渲染与评估

~~~powershell
python render.py -m .\output\pscd_vgg16 --iteration 30000 --skip_train
python metrics.py -m .\output\pscd_vgg16
~~~

结果保存在 results.json 与 per_view.json，包括 SSIM、PSNR 与 LPIPS。为保证可复现性，请固定数据版本、分辨率、--eval 划分和训练步数，并记录 GPU、驱动、CUDA、PyTorch 与 torchvision 版本以及 cfg_args。

完整参数说明、常见问题和批量评测限制见 [README_PSCD_zh.md](README_PSCD_zh.md)。使用前请阅读 [LICENSE.md](LICENSE.md)。
