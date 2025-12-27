# 1. 继承官方 Mask R-CNN 配置
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

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

        # 3. 插入 LWGA 模块
        plugins=[
            dict(
                cfg=dict(type='LWGAModule', split_size=2, kernel_size=3),
                position='after_conv3',
                stages=(True, True, True, True)
            )
        ]
    )
)