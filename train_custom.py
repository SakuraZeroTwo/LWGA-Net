# train_custom.py
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# ==========================================================
# 关键：导入自定义模块，触发注册机制！
# ==========================================================
try:
    from my_modules.eca_layer import ECALayer
    from my_modules.lwga_layer import LWGAModule

    print("Successfully imported custom modules: ECALayer, LWGAModule")
except ImportError as e:
    print(f"Error importing custom modules: {e}")


# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with custom modules')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 默认保存路径：work_dirs/配置文件名
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    import torch
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # 初始化 Meta 信息
    meta = dict()
    meta['config'] = cfg.pretty_text
    meta['exp_name'] = osp.basename(args.config)

    # 构建模型
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 打印简要信息
    print(f"Config used: {args.config}")
    print(f"Work directory: {cfg.work_dir}")
    print(f"Backbone Plugins: {cfg.model.backbone.plugins}")

    cfg.seed = args.seed
    # 开始训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=meta
    )


if __name__ == '__main__':
    main()