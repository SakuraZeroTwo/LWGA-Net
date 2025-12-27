# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import PLUGIN_LAYERS
#
#
# @PLUGIN_LAYERS.register_module()
# class LWGAModule(nn.Module):
#     """
#     LWGA: Lightweight Weak Global Attention Module
#     论文对应: Figure 1 - Integrating global and local info
#     """
#
#     def __init__(self, in_channels, split_size=2, kernel_size=3):
#         super(LWGAModule, self).__init__()
#
#         self.split_size = split_size  # 对应论文中的划分参数 (决定 P 的大小)
#         # P = split_size * split_size (例如 2x2 分割，P=4)
#         self.num_partitions = split_size * split_size
#
#         # -----------------------------------------------------------
#         # Step 1: 初始降维 (Integrate channel info)
#         # 论文原文: "integrate the channel info with a KxK convolutional filter"
#         # 既然是生成 Attention Map，通常会将 C 压缩为 1
#         # -----------------------------------------------------------
#         self.reduce_conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size,
#                                      padding=kernel_size // 2, bias=False)
#         self.bn = nn.BatchNorm2d(1)  # 加个 BN 训练更稳定
#
#         # -----------------------------------------------------------
#         # Step 2: 局部路径 (Local Path - DS Conv)
#         # 论文原文: "perform independent KxK convolution... stacking partitions"
#         # 我们对 Stack 后的 P 个通道做 Depthwise Conv
#         # -----------------------------------------------------------
#         self.local_conv = nn.Conv2d(
#             self.num_partitions,  # 输入通道 = 分区数量 P
#             self.num_partitions,  # 输出通道 = 分区数量 P
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#             groups=self.num_partitions,  # Groups=In_channels 意味着是 Depthwise Conv
#             bias=False
#         )
#
#         # -----------------------------------------------------------
#         # Step 3: 全局路径 (Global Path - FC)
#         # 论文原文: "flatten these means... use a fully connected network"
#         # -----------------------------------------------------------
#         self.global_fc = nn.Sequential(
#             nn.Linear(self.num_partitions, self.num_partitions // 2 + 1),  # 降维 (中间层)
#             nn.ReLU(),
#             nn.Linear(self.num_partitions // 2 + 1, self.num_partitions),  # 升维回 P
#             nn.Sigmoid()  # 生成 0~1 的权重
#         )
#
#         # 最终的激活函数
#         self.final_sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.shape
#
#         # [Step 1] 降维: (N, C, H, W) -> (N, 1, H, W)
#         feat = self.reduce_conv(x)
#         feat = self.bn(feat)
#
#         # [Step 2] Padding: 确保长宽能被 split_size 整除
#         # 论文原文: "pad ... to evenly partition"
#         pad_h = (self.split_size - h % self.split_size) % self.split_size
#         pad_w = (self.split_size - w % self.split_size) % self.split_size
#
#         # F.pad 参数顺序是 (左, 右, 上, 下)
#         feat_padded = F.pad(feat, (0, pad_w, 0, pad_h))
#
#         # 更新 padding 后的大小
#         h_pad, w_pad = feat_padded.shape[2], feat_padded.shape[3]
#
#         # [Step 3] Partition & Stack (关键步骤！)
#         # 我们要把图切成 split_size * split_size 块，并堆叠到通道维度
#         # 变换流程: (N, 1, H, W) -> (N, 1, s, h/s, s, w/s) -> (N, s*s, h/s, w/s)
#         sub_h = h_pad // self.split_size
#         sub_w = w_pad // self.split_size
#
#         # 1. 变形: 把大图拆成网格
#         feat_unfolded = feat_padded.view(n, 1, self.split_size, sub_h, self.split_size, sub_w)
#         # 2. 换位: 把网格维度(s, s)移到通道维度
#         feat_stacked = feat_unfolded.permute(0, 2, 4, 1, 3, 5).contiguous()
#         # 3. 拍扁: 合并通道 (N, P, sub_h, sub_w), 其中 P = s*s
#         feat_stacked = feat_stacked.view(n, self.num_partitions, sub_h, sub_w)
#
#         # ==================== 分支处理 ====================
#
#         # [Branch A] Local Attention (DS Conv)
#         local_att = self.local_conv(feat_stacked)
#
#         # [Branch B] Global Attention (FC)
#         # 1. 对每个分区求均值 -> (N, P, 1, 1)
#         global_pool = F.adaptive_avg_pool2d(feat_stacked, (1, 1))
#         # 2. 展平 -> (N, P)
#         global_vec = global_pool.view(n, self.num_partitions)
#         # 3. FC -> (N, P) -> (N, P, 1, 1)
#         global_att = self.global_fc(global_vec).view(n, self.num_partitions, 1, 1)
#
#         # [Fusion] 融合: 局部 * 全局
#         # 论文原文: "multiply the two types of attention weights"
#         # (N, P, sub_h, sub_w) * (N, P, 1, 1) -> 广播乘法
#         combined_att = local_att * global_att
#
#         # ================================================
#
#         # [Step 4] Reconstruction (还原)
#         # 逆操作: 把 P 个小图拼回一张大图
#         # (N, P, sub_h, sub_w) -> (N, s, s, 1, sub_h, sub_w)
#         att_unstacked = combined_att.view(n, self.split_size, self.split_size, 1, sub_h, sub_w)
#         # 换位回空间顺序: (N, 1, s, sub_h, s, sub_w)
#         att_restored = att_unstacked.permute(0, 3, 1, 4, 2, 5).contiguous()
#         # 拍扁回大图: (N, 1, H_pad, W_pad)
#         att_map = att_restored.view(n, 1, h_pad, w_pad)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class LWGAModule(nn.Module):
    def __init__(self, in_channels, split_size=2, kernel_size=3):
        super(LWGAModule, self).__init__()
        self.split_size = split_size
        self.num_partitions = split_size * split_size

        # Step 1: Reduce Dimension
        self.reduce_conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size,
                                     padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Step 2: Local Path (DS Conv)
        self.local_conv = nn.Conv2d(
            self.num_partitions, self.num_partitions,
            kernel_size=kernel_size, padding=kernel_size // 2,
            groups=self.num_partitions, bias=False
        )

        # Step 3: Global Path (FC)
        self.global_fc = nn.Sequential(
            nn.Linear(self.num_partitions, self.num_partitions // 2 + 1),
            nn.ReLU(),
            nn.Linear(self.num_partitions // 2 + 1, self.num_partitions),
            nn.Sigmoid()
        )
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        # Reduce
        feat = self.bn(self.reduce_conv(x))

        # Padding
        pad_h = (self.split_size - h % self.split_size) % self.split_size
        pad_w = (self.split_size - w % self.split_size) % self.split_size
        feat_padded = F.pad(feat, (0, pad_w, 0, pad_h))
        h_pad, w_pad = feat_padded.shape[2], feat_padded.shape[3]

        # Partition & Stack
        sub_h, sub_w = h_pad // self.split_size, w_pad // self.split_size
        feat_stacked = feat_padded.view(n, 1, self.split_size, sub_h, self.split_size, sub_w)
        feat_stacked = feat_stacked.permute(0, 2, 4, 1, 3, 5).contiguous().view(n, self.num_partitions, sub_h, sub_w)

        # Attention Calculation
        local_att = self.local_conv(feat_stacked)
        global_vec = F.adaptive_avg_pool2d(feat_stacked, (1, 1)).view(n, self.num_partitions)
        global_att = self.global_fc(global_vec).view(n, self.num_partitions, 1, 1)

        combined_att = local_att * global_att

        # Reconstruction & Crop
        att_restored = combined_att.view(n, self.split_size, self.split_size, 1, sub_h, sub_w)
        att_restored = att_restored.permute(0, 3, 1, 4, 2, 5).contiguous().view(n, 1, h_pad, w_pad)

        att_map = att_restored[:, :, :h, :w]  # Crop back to original size

        return identity * self.final_sigmoid(att_map)