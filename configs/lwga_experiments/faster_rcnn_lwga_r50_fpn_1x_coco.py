# 1. 继承官方的基础配置
# 注意：因为我们在 configs/lwga_experiment/ 下，所以要往上跳两级 (../faster_rcnn/...)
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        # 2. 这里的 ResNet 配置会覆盖基础配置里的 backbone
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),

        # =====================================================
        # 3. 核心修改：插入你的 LWGA 模块
        # =====================================================
        plugins=[
            dict(
                # cfg 里的参数对应 LWGAModule.__init__ 的参数
                # in_channels 会由 MMDetection 自动填入，不需要写
                cfg=dict(type='LWGAModule', split_size=2, kernel_size=3),

                # 插入位置：Bottleneck 的第3个卷积(conv3)之后，相加(residual add)之前
                # 这和 ECA-Net、CBAM 的官方插入位置一致
                position='after_conv3',

                # 在 ResNet 的所有 Stage (1,2,3,4) 都启用
                stages=(True, True, True, True)
            )
        ]
    )
)