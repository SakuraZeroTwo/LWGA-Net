import torch
import time
from mmcv import Config
from mmdet.models import build_detector

# ==================================================================
# 关键步骤：必须先导入这两个模块，否则 MMDetection 读不懂配置文件
# ==================================================================
try:
    from my_modules.eca_layer import ECALayer
    from my_modules.lwga_layer import LWGAModule

    print(">> 成功导入自定义模块 (ECA & LWGA)")
except ImportError as e:
    print(f"!! 导入模块失败: {e}")
    print("请确保 my_modules 文件夹里有 __init__.py，且路径正确")
    exit()


def run_test(config_path, model_name):
    print(f"\n{'=' * 20} 开始测试: {model_name} {'=' * 20}")
    print(f"读取配置文件: {config_path}")

    try:
        # 1. 读取配置
        cfg = Config.fromfile(config_path)

        # 2. 构建模型
        print("正在构建模型结构...")
        model = build_detector(cfg.model)

        # 3. 搬到 GPU (如果有)
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        print(f"运行设备: {device}")

        # 4. 造假数据 (模拟一张 3通道 800x800 的图片)
        # Batch Size = 1
        dummy_img = torch.randn(1, 3, 800, 800)
        if device == 'cuda':
            dummy_img = dummy_img.cuda()

        # 5. 前向传播测试 (Backbone + Neck + Head)
        print("正在进行前向传播 (Forward Pass)...")
        start_time = time.time()

        # extract_feat 会调用 backbone (包含你的 ECA/LWGA)
        feats = model.extract_feat(dummy_img)

        end_time = time.time()

        # 6. 验证输出
        print(f"✅ {model_name} 测试通过！")
        print(f"耗时: {end_time - start_time:.4f} 秒")
        print(f"输出特征层数量: {len(feats)} (通常 FPN 输出 5 层)")
        print(f"最深层特征图尺寸: {feats[-1].shape}")

    except Exception as e:
        print(f"❌ {model_name} 测试失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # -------------------------------------------------
    # 测试 1: ECA-Net (ResNet50)
    # -------------------------------------------------
    run_test(
        config_path='configs/eca_net/faster_rcnn_eca_r50_fpn_1x_coco.py',
        model_name='ECA-Net'
    )

    # -------------------------------------------------
    # 测试 2: LWGA-Net
    # -------------------------------------------------
    run_test(
        config_path='configs/lwga_experiments/faster_rcnn_lwga_r50_fpn_1x_coco.py',
        model_name='LWGA-Net'
    )