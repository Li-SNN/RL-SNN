
"""
@CreatedDate:   2022/04
@Author: lyh
"""
import math
from torch.distributions import Dirichlet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.activation_based import neuron, surrogate, encoding
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import torch

def complex_relu(input_r, input_i):
    return relu(input_r), relu(input_i)

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # temp = torch.abs(1 - torch.abs(torch.arcsin(input))) < 0.7
        temp = (1 / 2.5) * torch.sign(abs(input) < 2.5)
        return grad_input * temp.float()

class MultiStepLIFNode(nn.Module):
    def __init__(self, tau=1.0, detach_reset=True, backend='cupy', v_threshold=1.0):
        super().__init__()
        self.spike_fn = Surrogate_BP_Function.apply

        # 将 tau 初始化为一个可学习的参数。
        # 初始值设为 1.0 时，经过 sigmoid 约为 0.73，与你之前写死的 0.7 非常接近，利于平滑过渡。
        self.tau = nn.Parameter(torch.tensor([tau], dtype=torch.float))
        self.v_threshold = nn.Parameter(torch.tensor([1.0]))

        # self.v_threshold = v_threshold
        self.detach_reset = detach_reset

        # 定义脉冲计数器
        self.spike_count = 0  # 记录发放的脉冲总数
        self.neuron_count = 0  # 记录经过该节点的神经元总数(B * T * C * H * W)

    def forward(self, x_seq: torch.Tensor):

        spike_seq = []
        # 初始化膜电位
        mem = torch.zeros_like(x_seq[0])

        # 将自适应 tau 限制在 (0, 1) 区间，作为膜电位的衰减因子 (decay)
        decay = torch.sigmoid(self.tau)

        for t in range(x_seq.shape[0]):
            # 膜电位更新：利用自适应的衰减因子保留历史信息，并整合当前输入
            mem = decay * mem + x_seq[t]

            # 脉冲发放：当膜电位超过阈值时发放脉冲
            mem_thr = mem - self.v_threshold
            x = self.spike_fn(mem_thr)

            # 膜电位重置 (Soft Reset 机制)：
            # 减去发放脉冲带走的电位 (乘以阈值保证量纲一致)
            # 如果 detach_reset 为 True，可以切断重置过程的梯度，避免复杂的梯度传播计算
            if self.detach_reset:
                mem = mem - (x * self.v_threshold).detach()
            else:
                mem = mem - x * self.v_threshold

            spike_seq.append(x.unsqueeze(0))

        # 拼接时间维度，返回 [T, B, C, H, W] 的脉冲张量
        spike_seq = torch.cat(spike_seq, 0)

        # 在非训练模式下统计脉冲发放率
        if not self.training:
            # spike_seq 包含 0 和 1，求和即为脉冲总数
            self.spike_count += spike_seq.detach().sum().item()
            # numel() 返回张量中元素的总个数
            self.neuron_count += spike_seq.numel()

        return spike_seq

    # 提供重置和获取状态的方法
    def get_firing_rate(self):
        if self.neuron_count == 0:
            return 0.0
        return self.spike_count / self.neuron_count

    def reset_stats(self):
        self.spike_count = 0
        self.neuron_count = 0

class TTFS_SpectralEncoder(nn.Module):
    def __init__(self, T, beta=10.0):
        super().__init__()
        self.T = T
        self.beta = beta

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        x = x.squeeze(1)
        t_fire = (1.0 - x) * (self.T - 1)
        t = torch.arange(self.T, device=x.device).view(self.T, 1, 1)
        s = torch.sigmoid(self.beta * (t_fire.unsqueeze(0) - t))
        spikes = s * (1 - s.roll(1, dims=0))
        spikes[0] = s[0]
        return spikes.float()

class TemporalDownsampler(nn.Module):

    def __init__(self, t_high, t_low):
        super().__init__()
        assert t_high % t_low == 0, "T_high must be divisible by T_low"
        self.stride = t_high // t_low

    def forward(self, x):
        # x shape: [T_high, B, C, ...]
        t_high, b = x.shape[0], x.shape[1]
        other_dims = x.shape[2:]

        # 重新排列，将时间维度拆分
        # [T_low, Stride, B, C, ...]
        x = x.view(-1, self.stride, b, *other_dims)

        # 在 Stride 维度取 Max (保持脉冲的显著性)
        # 只要这一小段间隔内有脉冲发生，就认为压缩后的时间步有脉冲
        x_compressed, _ = torch.max(x, dim=1)

        return x_compressed  # [T_low, B, C, ...]

class SpectralSpikingBranch(nn.Module):
    def __init__(self, in_bands, T):
        super().__init__()

        self.T = T
        self.TTFS = TTFS_SpectralEncoder(in_bands)
        # self.TTFS = TTFS_SpectralEncoder(self.T)

        # 时域下采样模块
        self.downsampler = TemporalDownsampler(in_bands, self.T)

        self.conv1 = nn.Conv1d(in_bands, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.lif1 = MultiStepLIFNode()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.lif2 = MultiStepLIFNode()

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.lif3 = MultiStepLIFNode()
        # 新增：用于存储所有 Batch 的脉冲统计数据
        self.all_spike_counts = None

    def forward(self, x):
        # TTFS之前需要先做归一化
        # x = (x - x.min()) / (x.max() - x.min())

        B, C, H, W = x.shape


        # 1️⃣ 空间平均
        x = x.mean(dim=[2,3])  # [B,C]
        # x = x.mean(dim=[3,4])  # [B,C]
        # x = x[:, :, H // 2, W // 2]  # 提取中心光谱 [B, C]
        # x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # 2️⃣ TTFS编码
        x = self.TTFS(x)  # [T,B,C]

        # --- 时域压缩 (High T -> Low T) ---
        # spikes_low: [T_low, B, C]
        spikes_low = self.downsampler(x)

        # 3️⃣ reshape给Conv1d
        x = spikes_low.reshape(self.T * B, C, 1)
        # x = spikes_low.reshape(self.T * B, 1, C)
        # x = x.reshape(self.T * B, C, 1)

        # Block1
        x = self.conv1(x)
        # x = self.bn1(x)
        x = x.reshape(self.T, B, 32, -1)
        x = self.lif1(x)

        # Block2
        x = x.reshape(self.T * B, 32, -1)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = x.reshape(self.T, B, 64, -1)
        x = self.lif2(x)

        # Block3
        x = x.reshape(self.T * B, 64, -1)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = x.reshape(self.T, B, 128, -1)
        x = self.lif3(x)

        x = x.mean(dim=3)  # 光谱池化
        # 加入RL之后时间平均才去除
        # x = x.mean(dim=0)  # 时间平均

        return x
class SpatialSpikingBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # --- Layer 1 ---
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lif1 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)
        # 对应 LiDAR x1: in_channels -> Spatial x1: in_channels
        # self.guide1 = SpikingFeatureGuidance(in_channels, in_channels)

        # --- Layer 2 ---
        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)
        # 对应 LiDAR x2: 32 -> Spatial x2: 64
        self.guide2 = SpikingFeatureGuidance(64, 64)

        # --- Layer 3 ---
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.lif5 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)
        # 对应 LiDAR x3: 64 -> Spatial x3: 128
        self.guide3 = SpikingFeatureGuidance(128, 128)

        # --- Layer 4 ---
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        # self.lif6 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.lif7 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1)
        # 对应 LiDAR x4: 128 -> Spatial x4: 128
        # self.guide4 = SpikingFeatureGuidance(128, 128)

    def forward(self, x, lidar_feats, T):
        B, C, H, W = x.shape
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        self.T = x.shape[0]

        # 解包 LiDAR 四层特征
        l1, l2, l3 = lidar_feats

        # --- Layer 1 ---
        x1 = self.bn1(self.conv1(x.flatten(0, 1))).reshape(self.T, B, -1, H, W)
        # x1 = self.conv1(x.flatten(0, 1)).reshape(self.T, B, -1, H, W)
        x1_lif = self.lif1(x1)
        # x1 = self.guide1(x1, l1)  # 使用 l1 引导

        # --- Layer 2 ---
        x2 = self.bn2(self.conv2(x1_lif.flatten(0, 1))).reshape(self.T, B, -1, H, W)
        # x2_Res = torch.cat([x1,x2],dim=1)
        x2_lif = self.lif2(x2)


        # --- Layer 3 ---
        x3 = self.bn3(self.conv3(x2_lif.flatten(0, 1))).reshape(self.T, B, -1, H, W)
        # x3_Res = x3+x2
        x3_lif = self.lif3(x3)+x2_lif

        # x3_guide = self.guide2(x3_lif, l2)  # 使用 l2 引导

        x4 = self.bn4(self.conv4(x3_lif.flatten(0, 1))).reshape(self.T, B, -1, H, W)
        x4_lif = self.lif4(x4)

        x5 = self.bn5(self.conv5(x4_lif.flatten(0, 1))).reshape(self.T, B, -1, H, W)
        # x5_Res = x5+x4
        x5_lif = self.lif5(x5)+x4_lif

        x6 = self.guide3(x5_lif, l3)  # 使用 l3 引导

        # x7 = self.conv6(x6.flatten(0, 1)).reshape(self.T, B, -1, H, W)
        # x7_lif = self.lif6(x7)

        # x8 = self.conv7(x6.flatten(0, 1)).reshape(self.T, B, -1, H, W)
        # # x7_Res = x7 + x6
        # x8_lif = self.lif7(x8)

        # --- Layer 4 ---

        # x4 = self.guide4(x4, l4)  # 使用 l4 引导

        return x6

class LiDARSpikingBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy',v_threshold=1)
        # self.lif1 = MultiStepLIFNode(64)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy',v_threshold=1)
        # self.lif2 = MultiStepLIFNode(128)
        # self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy',v_threshold=1)
        # self.lif3 = MultiStepLIFNode(128)


    def forward(self, x, T):
        B, C, H, W = x.shape
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        self.T = x.shape[0]

        x1 = self.bn1(self.conv1(x.flatten(0, 1)))
        # x1 = self.conv1(x.flatten(0, 1))
        x1 = x1.reshape(self.T, B, -1, H, W)
        # x1 = self.lif1(x1,0.7,1.0)
        x1 = self.lif1(x1)

        x2 = self.bn2(self.conv2(x1.flatten(0, 1)))
        # x2 = self.conv2(x1.flatten(0, 1))
        x2 = x2.reshape(self.T, B, -1, H, W)
        # x2 = self.lif2(x2,0.7,1.0)
        x2 = self.lif2(x2)

        x3 = self.bn3(self.conv3(x2.flatten(0, 1)))
        # x3 = self.conv3(x2.flatten(0, 1))
        x3 = x3.reshape(self.T, B, -1, H, W)
        # x3 = self.lif3(x3,0.7,1.0)
        x3 = self.lif3(x3)

        # x4 = self.conv4(x3.flatten(0, 1))
        # x4 = x4.reshape(self.T, B, -1, H, W)
        # # x3 = self.lif3(x3,0.7,1.0)
        # x4 = self.lif4(x4)

        return [x1,x2,x3]


class SpikingFeatureGuidance(nn.Module):
    """
    脉冲特征引导模块 (SFGM)
    利用 LiDAR 的脉冲特征作为先验，生成空间门控掩膜，引导 HSI 空间特征提取。
    """

    def __init__(self, in_channels_lidar, in_channels_spat):
        super().__init__()
        # 跨模态通道对齐：将 LiDAR 的通道数映射为与空间特征一致，核大小为1
        self.conv_align = nn.Conv2d(in_channels_lidar, in_channels_spat, kernel_size=1)
        # 生成门控掩膜的 LIF 神经元
        self.lif_gate = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1.0)
        # 融合后的再发放 LIF 神经元
        self.lif_out = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy', v_threshold=1.0)

    def forward(self, x_spat, x_lidar):
        """
        x_spat: [T, B, C_spat, H, W]
        x_lidar: [T, B, C_lidar, H, W]
        """
        T, B, C, H, W = x_spat.shape

        # LiDAR 特征对齐与掩膜生成
        l_flat = x_lidar.flatten(0, 1)  # [T*B, C_lidar, H, W]
        gate = self.conv_align(l_flat)
        gate = gate.reshape(T, B, -1, H, W)

        # 产生 0 或 1 的脉冲门控掩膜 (Spiking Mask)
        spike_gate = self.lif_gate(gate)

        #  脉冲域引导 (Spiking Modulation)
        # 脉冲的乘法相当于逻辑“与”(AND)操作。
        # 当 LiDAR 认为这里有结构边界(spike=1)，保留 HSI 空间特征；否则(spike=0)抑制。
        # 加上残差结构 x_spat，保证特征不会因为过度抑制而消失
        guided_spat = x_spat * spike_gate + x_spat

        # 稳态输出
        # 通过最后一层 LIF，重新整合膜电位并激发新的脉冲
        out = self.lif_out(guided_spat)

        return out
class SpikingClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 加入 LayerNorm 修复权重缩放导致的脉冲稀释
        self.norm = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)
        self.drop = nn.Dropout(0.1)
        # self.drop = nn.Dropout(0)
        # 使用自定义的 MultiStepLIFNode 保持动力学一致性
        self.lif = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy',v_threshold=1)

    def forward(self, x_seq):
        # x_seq: [T, B, C]
        # 1. 线性映射 [T, B, C] -> [T, B, Classes]
        T, B, C = x_seq.shape
        x_seq = self.norm(x_seq.flatten(0, 1)).reshape(T, B, C)
        x_seq = self.drop(x_seq)
        x_seq = self.fc(x_seq.flatten(0, 1)).reshape(T, B, -1)

        # 脉冲激发 [T, B, Classes]
        # spike_seq = self.lif(x_seq)

        # 计算 T 个时间步内的平均脉冲发放率
        # return spike_seq.mean(dim=0)
        return x_seq.mean(dim=0)

class SpikingPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=128):
        super().__init__()
        #加入 LayerNorm 稳定输入
        self.norm = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lif1 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy',v_threshold=1)
        self.fc_delta = nn.Linear(hidden_dim, action_dim)

        # 将方差设为可学习参数，但初始值极小，限制探索步长
        self.std = nn.Parameter(torch.ones(1, action_dim) * 0.05)

    # def forward(self, spike_seq):
    def forward(self, spike_seq,boundary):
        T, B, C = spike_seq.shape
        # 对输入进行标准化
        x = self.norm(spike_seq.flatten(0, 1)).reshape(T, B, C)
        x = self.fc1(x.flatten(0, 1)).reshape(T, B, -1)
        x = self.lif1(x)
        feat = x.mean(dim=0)


        # 针对Houston2013数据集的修改
        mu = 1/3 + torch.tanh(self.fc_delta(feat)) * boundary

        # 增加探索噪声的下限，Houston 的复杂性需要更强的随机性
        # std = torch.clamp(self.std * (0.1 / 0.1), 0.02, 0.2)

        # 基础权重约等于 0.33，策略网络只做上下 0.1 的微调 (使用 Tanh 限制范围)
        # mu = 0.333 + torch.tanh(self.fc_delta(feat)) * 0.3
        #
        # # 限制标准差在极小范围内，防止探索导致精度崩塌
        # std = torch.clamp(self.std * (boundary / 0.1), min=1e-3, max=0.2)
        std = torch.clamp(self.std * (boundary / 0.1), min=1e-3, max=0.2)
        # std = torch.clamp(self.std, min=1e-3, max=0.08)
        dist = torch.distributions.Normal(mu, std)

        if self.training:
            action = dist.rsample()  # 重参数化采样
        else:
            action = mu  # 测试时使用确定的均值

        # Softmax 归一化为最终的注意力权重 w
        w = torch.softmax(action, dim=1)

        # 计算 log_prob 和 entropy 供强化学习使用
        log_prob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)

        return w, log_prob, entropy

class ComplexNet(nn.Module):
    def __init__(self, num_steps, input_dim, num_classes):
        super(ComplexNet, self).__init__()
        self.T = num_steps

        # 脉冲特征提取分支
        # self.branch_lidar = LiDARSpikingBranch(1)
        self.branch_lidar = LiDARSpikingBranch(4)
        # self.branch_lidar = LiDARSpikingBranch(2)
        self.branch_spec = SpectralSpikingBranch(input_dim, num_steps)
        self.branch_spat = SpatialSpikingBranch(input_dim)


        # 脉冲辅助分类器：利用频率解码
        self.aux_spec = SpikingClassifier(128, num_classes)
        self.aux_spat = SpikingClassifier(128, num_classes)
        self.aux_lidar = SpikingClassifier(128, num_classes)

        self.policy = SpikingPolicyNet(state_dim=128 + 128 + 128, action_dim=3)
        self.classifier = SpikingClassifier(384, num_classes)  # 总维度 128+128+64=320

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)

    # def forward(self, hsi, lidar):
    def forward(self, hsi, lidar,boundary):
        device = next(self.parameters()).device

        hsi = hsi.to(device)
        lidar = lidar.to(device)

        # 提取三分支脉冲序列 (保持时间维 [T, B, C])
        s_spec = self.branch_spec(hsi)  # [T, B, 128]

        # 先计算 LiDAR 分支，获取多尺度特征
        lidar_feats = self.branch_lidar(lidar, self.T)
        # 取 LiDAR 的最后一层特征参与最终融合与分类 (lidar_feats[-1] 即 x3)
        s_lida = lidar_feats[-1].mean(dim=[3, 4])  # [T, B, 64 -> 128]

        # 将 LiDAR 多尺度特征传入 Spatial 分支进行空间引导
        s_spat = self.branch_spat(hsi, lidar_feats, self.T).mean(dim=[3, 4])  # [T, B, 128]

        # 计算辅助分支 Logits (用于 RL 的 Reward 计算)
        l_spec = self.aux_spec(s_spec)
        l_spat = self.aux_spat(s_spat)
        l_lida = self.aux_lidar(s_lida)

        # 策略网络推断权重
        state_spikes = torch.cat([s_spec, s_spat, s_lida], dim=2)  # [T, B, 320]
        w, log_prob, entropy = self.policy(state_spikes,boundary)  # w shape: [B, 3]

        # 加权融合 (每一个 batch 中的样本权重都不同)
        # w[:, 0:1] 对应光谱分支，w[:, 1:2] 对应空间分支...
        f_spec = w[:, 0:1].unsqueeze(0) * s_spec
        f_spat = w[:, 1:2].unsqueeze(0) * s_spat
        f_lida = w[:, 2:3].unsqueeze(0) * s_lida

        fusion_seq = torch.cat([f_spec, f_spat, f_lida], dim=2)  # [T, B, 320]

        out = self.classifier(fusion_seq)
        return out, log_prob, (l_spec, l_spat, l_lida), entropy,w

