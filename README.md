# LWGA-Net: Lightweight Weak Global Attention (轻量级弱全局注意力网络)

本项目是论文 **LWGA-Net** 的官方 PyTorch 实现代码。
基于 **MMDetection v2.28.2** 框架开发，并包含了 **ECA-Net** 的完整复现代码以供公平对比。

## 项目架构
- **核心模块**:
  - `my_modules/lwga_layer.py`: LWGA 模块实现（本文提出的方法）。
  - `my_modules/eca_layer.py`: ECA 模块实现（对比基准）。
- **配置文件**:
  - `configs/lwga_experiments/`: LWGA 相关的实验配置。
  - `configs/eca_net/`: ECA 相关的实验配置。

---

## 环境安装 

⚠️ **注意：** 本项目依赖特定版本的 PyTorch (1.x) 和 NumPy (1.x)。请务必严格按照以下顺序安装，否则会出现版本不兼容报错。

### 1. 创建虚拟环境
推荐使用 Python 3.10。
```bash
conda create -n lwga python=3.10 -y
conda activate lwga
```

### 2. 安装 PyTorch (推荐 1.13.1)
虽然显卡驱动可能支持 CUDA 12.x，但为了兼容 MMDetection v2，请安装适配 CUDA 11.7 的版本（向下兼容）。
code
```Bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
### 3. 安装 MMCV-Full
必须安装 1.7.x 版本，不要安装 2.x 版本。

``` Bash
pip install -U openmim
mim install mmcv-full==1.7.2
```
### 4. 安装项目依赖
这一步会自动安装安全的 NumPy (<2.0) 和 OpenCV 版本。

```Bash
pip install -r requirements.txt
```
### 5. 编译安装 MMDetection
使用 --no-build-isolation 避免新版 pip 的构建隔离问题。

``` Bash
pip install -v -e . --no-build-isolation
```
## 数据集与权重准备
由于 GitHub 限制，请手动准备 MS COCO 2017 数据集和 ResNet-50 预训练权重。
目录结构需保持如下：

```code
Text
LWGA-Net/
├── checkpoints/
│   └── resnet50-19c8e357.pth   <-- 请从 PyTorch/TorchVision 官方下载
├── data/
│   └── coco/                   <-- COCO 数据集
│       ├── annotations/
│       │   ├── instances_train2017.json
│       │   └── instances_val2017.json
│       ├── train2017/
│       └── val2017/
├── configs/
├── my_modules/
├── train_custom.py
└── ...
```

##  开始训练
本项目提供了一个自定义启动脚本 train_custom.py，它会自动注册 LWGA 和 ECA 模块，无需修改底层源码。
### 实验 1：训练 LWGA-Net (Ours)
使用 ResNet-50 + Faster R-CNN + LWGA 模块：

```Bash
python train_custom.py configs/lwga_experiments/faster_rcnn_lwga_r50_fpn_1x_coco.py
```
### 实验 2：训练 ECA-Net (Baseline)
使用 ResNet-50 + Faster R-CNN + ECA 模块：
code
```Bash
python train_custom.py configs/eca_net/faster_rcnn_eca_r50_fpn_1x_coco.py
```
### 其他检测器
若需测试 Mask R-CNN 或 RetinaNet，只需更换配置文件路径即可：
```bash
configs/lwga_experiments/mask_rcnn_lwga_r50_fpn_1x_coco.py
configs/lwga_experiments/retinanet_lwga_r50_fpn_1x_coco.py
```
## 实验结果 
```
Method	 Backbone	Detector	mAP (val)	Params	FLOPs
ECA-Net	 ResNet-50	Faster R-CNN	...	...	...
LWGA-Net ResNet-50	Faster R-CNN	...	...	...
```
## 🤝 致谢
本项目基于 MMDetection 框架开发。