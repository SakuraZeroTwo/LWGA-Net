# 继承基础的 RetinaNet 配置
_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),

        # 插入 ECA 插件
        plugins=[
            dict(
                cfg=dict(type='ECALayer', gamma=2, b=1),
                position='after_conv3',
                stages=(True, True, True, True)
            )
        ]
    )
)