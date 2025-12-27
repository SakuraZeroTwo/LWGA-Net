# 继承基础的Faster R-CNN配置
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

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
        # 插入ECA插件

        plugins=[
            dict(
                cfg=dict(type='ECALayer', gamma=2, b=1),
                position='after_conv3', # 插入位置：在 Bottleneck 的最后一个卷积之后
                stages=(True, True, True, True)# 在 ResNet 的所有 4 个 Stage 都启用
            )
        ]
    )
)
