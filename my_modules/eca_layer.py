import math
import torch
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS

@PLUGIN_LAYERS.register_module()
class ECALayer(nn.Module):
    """
        ECA-Net 核心模块
        论文题目: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    """
    def __init__(self,in_channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        self.in_channels = in_channels

        # 自适应计算卷积核大小
        # 公式: k = |(log2(C) + b) / gamma|_odd
        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        # 定义1D卷积
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding = k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.avg_pool(x)  # (N, C, 1, 1)

        # 调整维度以适应1D卷积：(N,C,1,1) -> (N,1,C)
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = self.sigmoid(y)

        # 还原维度：(N,1,C) -> (N,C,1,1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 广播相乘
        return x * y.expand_as(x)