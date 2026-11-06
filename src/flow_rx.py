"""Flow-Rx：叠加导频 OFDM 的条件流匹配接收机。

本文件演示 Conditional Flow Matching 的基本流程：
1. 从 CDL 数据集中读取 SISO 复数信道；
2. 每个样本等概率选择 QPSK、16QAM 或 64QAM，生成叠加 QPSK 导频的接收信号；
3. 在高斯噪声和数据之间构造线性插值路径；
4. 混合训练同一个 DiT 预测数据终点，并转换为 Flow Matching 速度；
5. 使用最优模型分别测试三种调制方式的 BER–SNR 曲线。

当前示例使用归一化后的 CDL 复数信道生成接收信号，以三种调制的混合数据
为目标分布。每个样本内所有时频符号使用同一种调制，网络输入由插值状态、
接收信号和固定导频的实部与虚部组成，不额外输入调制类型。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from typing import Any

from pathlib import Path

import swanlab
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm.auto import tqdm

if __package__:
    from .throughput_metrics import block_throughput_metrics, count_successful_blocks
else:
    from throughput_metrics import block_throughput_metrics, count_successful_blocks


MODULATIONS = ("QPSK", "16QAM", "64QAM")


class ChannelDataset(Dataset[torch.Tensor]):
    """读取归一化后的 CDL 复数信道数据。

    Args:
        path: .pt 格式信道数据文件路径。

    Returns:
        ChannelDataset: 每次索引返回一个复数 SISO 信道张量。

    实现流程是加载形状为 [样本数, 时间, 接收天线, 发射天线, 子载波]
    的复数信道张量，固定选择第 0 个接收天线和第 0 个发射天线，丢弃
    接收天线和发射天线维度以得到 SISO 数据 [样本数, 时间, 子载波]。
    信道统一为 complex64，实部与虚部均保留 float32 精度，并检查非有限值。
    随后将每个复数元素拆分为实部和虚部，并调整为
    [样本数, 2, 时间, 子载波]，其中第 1 维的两个通道分别表示实部和虚部。
    """

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"找不到信道数据文件: {path}")
        channel = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(channel, torch.Tensor) or not channel.is_complex():
            raise TypeError("信道文件必须包含复数 torch.Tensor")
        if channel.ndim < 2:
            raise ValueError("信道张量至少需要包含样本维和特征维")
        if channel.ndim != 5:
            raise ValueError(
                "信道张量必须是 [样本, 时间, 接收天线, 发射天线, 子载波] 五维张量"
            )
        # 选择一个接收天线和一个发射天线，保留第 0、1、4 维实现 SISO。
        if any(size == 0 for size in channel.shape):
            raise ValueError("信道张量的所有维度必须非空")
        channel = channel[:, :, 0, 0, :].to(torch.complex64)
        if not torch.isfinite(channel).all():
            raise ValueError("信道数据包含 NaN 或 Inf")
        # view_as_real 后形状为 [样本, 时间, 子载波, 2]，将实虚部移到通道维。
        self.channel = channel
        self.samples = torch.view_as_real(channel).to(torch.float32).permute(0, 3, 1, 2)
        self.path = path

    def __len__(self) -> int:
        """返回数据集大小。

        Returns:
            int: 数据集中的样本数量。
        """
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """读取一个目标分布样本。

        Args:
            index: 样本索引。

        Returns:
            torch.Tensor: 形状为 [时间, 子载波] 的复数 SISO 信道样本。

        实现流程是检查索引范围，然后返回预先生成的对应样本。
        """
        if index < 0 or index >= self.samples.shape[0]:
            raise IndexError("样本索引超出范围")
        return self.channel[index]


def generate_qpsk_signal(
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """生成单位功率的随机 QPSK 复数信号。

    Args:
        shape: 输出信号形状。
        generator: 可选的随机数生成器。

    Returns:
        torch.Tensor: 复数 QPSK 信号，平均功率约为 1。

    实现流程是分别随机生成实部和虚部符号，并组合为四种等概率的
    QPSK 星座点。该函数用于生成数据和固定导频。
    """
    real = torch.randint(0, 2, shape, generator=generator).to(torch.float32) * 2 - 1
    imag = torch.randint(0, 2, shape, generator=generator).to(torch.float32) * 2 - 1
    return torch.complex(real, imag) / 2**0.5


def generate_data_signal(
    shape: tuple[int, ...],
    modulation: str = "QPSK",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """生成理论平均功率为 1 的随机数据符号。

    Args:
        shape: 输出复数数据的形状。
        modulation: 数据调制方式，支持 QPSK、16QAM 和 64QAM。
        generator: 可选的随机数生成器。

    Returns:
        torch.Tensor: 指定形状的 complex64 数据符号。

    实现流程是保持原有 QPSK 生成方式；QAM 则在实虚两轴独立等概率
    采样奇数电平，并除以理论平均符号能量的平方根，使平均功率为 1。
    导频由独立的 QPSK 生成函数生成，不受本参数影响。
    """
    if modulation == "QPSK":
        return generate_qpsk_signal(shape, generator)
    if modulation not in {"16QAM", "64QAM"}:
        raise ValueError("modulation 必须是 QPSK、16QAM 或 64QAM")
    levels = 4 if modulation == "16QAM" else 8
    real = 2 * torch.randint(levels, shape, generator=generator).float() - (levels - 1)
    imag = 2 * torch.randint(levels, shape, generator=generator).float() - (levels - 1)
    return torch.complex(real, imag) / math.sqrt(2 * (levels**2 - 1) / 3)


def generate_mixed_data_signal(
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """按样本等概率混合三种调制方式，生成单位理论平均功率的数据。

    Args:
        shape: 输出形状，第一维为批次，其余维度为单个样本的符号布局。
        generator: 可选的 CPU 随机数生成器，同时控制调制选择与符号采样。

    Returns:
        torch.Tensor: CPU 上形状为 shape 的 complex64 数据，每个样本内
            所有符号使用同一种调制，样本间独立等概率选择三种调制。

    实现流程是为每个样本随机选择一个调制索引，按索引分组生成随机符号，
    再填回原始样本位置。不同调制均采用各自的理论能量归一化。
    """
    if not shape or any(size < 1 for size in shape):
        raise ValueError("shape 必须包含批次维，且各维度必须为正数")
    modulation_indices = torch.randint(len(MODULATIONS), (shape[0],), generator=generator)
    data = torch.empty(shape, dtype=torch.complex64)
    for index, modulation in enumerate(MODULATIONS):
        mask = modulation_indices == index
        count = int(mask.sum().item())
        if count:
            data[mask] = generate_data_signal((count, *shape[1:]), modulation, generator)
    return data


def demodulate_bits(symbols: torch.Tensor, modulation: str = "QPSK") -> torch.Tensor:
    """将复数符号按最近电平判决转换为 Gray 编码比特。

    Args:
        symbols: 任意形状的复数符号或连续估计值。
        modulation: 数据调制方式，支持 QPSK、16QAM 和 64QAM。

    Returns:
        torch.Tensor: 布尔比特张量，形状为 [*symbols.shape, 每符号比特数]，
            实部比特在前、虚部比特在后，每轴按高位到低位排列。

    实现流程是对 QPSK 保持原有正负判决；对 QAM 将两轴估计值还原到
    未归一化电平，判决最近电平并截断到星座边界，再将电平索引转换为
    Gray 码。相邻电平仅相差一个比特，用于计算 BER。
    """
    if modulation == "QPSK":
        return torch.stack((symbols.real > 0, symbols.imag > 0), dim=-1)
    if modulation not in {"16QAM", "64QAM"}:
        raise ValueError("modulation 必须是 QPSK、16QAM 或 64QAM")
    levels = 4 if modulation == "16QAM" else 8
    bits_per_axis = 2 if levels == 4 else 3
    components = torch.stack((symbols.real, symbols.imag), dim=-1)
    indices = ((components * math.sqrt(2 * (levels**2 - 1) / 3) + levels - 1) / 2)
    indices = indices.round().clamp(0, levels - 1).long()
    gray = indices ^ (indices >> 1)
    shifts = torch.arange(bits_per_axis - 1, -1, -1, device=symbols.device)
    return ((gray.unsqueeze(-1) >> shifts) & 1).bool().flatten(-2)


def generate_fixed_pilot(
    time_steps: int,
    subcarriers: int,
    seed: int,
) -> torch.Tensor:
    """生成固定的 QPSK 时频导频图。

    Args:
        time_steps: 时间维度长度。
        subcarriers: 频率维度长度。
        seed: 导频专用随机种子。

    Returns:
        torch.Tensor: 形状为 [时间, 子载波] 的固定复数 QPSK 导频。

    实现流程是使用固定种子创建独立随机生成器，再生成一次 QPSK
    时频图；后续训练和验证调用均使用同一个导频张量。
    """
    generator = torch.Generator().manual_seed(seed)
    return generate_qpsk_signal((time_steps, subcarriers), generator)


def generate_received_signal(
    channel: torch.Tensor,
    data: torch.Tensor,
    pilot: torch.Tensor,
    pilot_power_factor: float,
    snr_db: float,
) -> torch.Tensor:
    """根据叠加导频 OFDM 模型生成接收信号。

    Args:
        channel: 复数 SISO 信道，形状为 [batch_size, time, frequency]。
        data: 复数数据符号，形状为 [batch_size, time, frequency]。
        pilot: 固定复数 QPSK 导频，形状为 [time, frequency]。
        pilot_power_factor: 导频功率因子。
        snr_db: 加性复高斯噪声的信噪比，单位为 dB。

    Returns:
        torch.Tensor: 加噪后的复数接收信号，形状为
        [batch_size, time, frequency]。

    实现流程是按照叠加导频公式生成发射信号，再进行逐元素信道传输，
    最后根据接收干净信号功率加入指定 SNR 的复高斯噪声。
    """
    transmit = (
        pilot_power_factor**0.5 * pilot.unsqueeze(0)
        + (1.0 - pilot_power_factor) ** 0.5 * data
    )
    clean_received = transmit * channel
    signal_power = torch.mean(torch.abs(clean_received) ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.sqrt(noise_power / 2.0) * torch.complex(
        torch.randn_like(clean_received.real),
        torch.randn_like(clean_received.real),
    )
    return clean_received + noise


class FourierTimeEmbedding(nn.Module):
    """生成时间变量的 Fourier 特征。

    Args:
        num_frequencies: 使用的 Fourier 频率数量。
        max_frequency: 最高 Fourier 频率。

    Returns:
        FourierTimeEmbedding: 将 [批次, 1] 的时间输入编码为
        [批次, 2 * num_frequencies] 的正弦余弦特征。

    实现流程是使用对数均匀分布的固定频率对时间变量进行正弦和余弦
    映射，使模型能够表达不同尺度的时间变化。
    """

    def __init__(self, num_frequencies: int, max_frequency: float = 16.0) -> None:
        super().__init__()
        if num_frequencies < 1 or max_frequency <= 0:
            raise ValueError("num_frequencies 必须为正数，max_frequency 必须大于 0")
        frequencies = torch.exp(
            torch.linspace(0.0, torch.log(torch.tensor(max_frequency)), num_frequencies)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """将时间值编码为 Fourier 特征。

        Args:
            time: 时间输入，形状为 [batch_size, 1]。

        Returns:
            torch.Tensor: Fourier 特征，形状为
            [batch_size, 2 * num_frequencies]。
        """
        time = time.float()
        frequencies = self.frequencies.float()
        angles = 2.0 * torch.pi * time * frequencies

        return torch.cat(
            (torch.sin(angles), torch.cos(angles)),
            dim=-1,
        )


class SwiGLU(nn.Module):
    """实现使用 SiLU 门控的前馈网络。

    Args:
        hidden_dim: 输入和输出 Token 的隐藏维度。
        intermediate_dim: 门控前馈层的中间维度。

    Returns:
        SwiGLU: 输入和输出形状均为 [..., hidden_dim] 的前馈模块。

    实现流程是将输入投影为门控分支和值分支，使用 SiLU 激活门控分支，
    再与值分支逐元素相乘并投影回隐藏维度。
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_dim, 2 * intermediate_dim)
        self.output_projection = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """计算 SwiGLU 前馈输出。

        Args:
            tokens: 输入 Token，最后一维大小为 hidden_dim。

        Returns:
            torch.Tensor: 经过门控变换后的 Token，形状与输入相同。
        """
        gate, value = self.input_projection(tokens).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class DiTBlock(nn.Module):
    """实现带时间条件调制的 DiT Transformer Block。

    Args:
        hidden_dim: Token 隐藏维度。
        num_heads: 多头自注意力头数。
        mlp_ratio: SwiGLU 前馈层中间维度相对于 hidden_dim 的倍率。

    Returns:
        DiTBlock: 对时频 Token 进行全局建模的 Transformer 模块。

    实现流程是先用时间嵌入生成自适应 LayerNorm 的缩放、偏移和门控参数，
    再依次计算自注意力分支和 SwiGLU 前馈分支，并通过残差连接更新 Token。
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden_dim)
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # AdaLN-Zero：初始时调制和残差门控均为零，使每个 Block 初始为恒等映射。
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """根据时间条件更新时频 Token。

        Args:
            tokens: 输入 Token，形状为 [batch_size, token_count, hidden_dim]。
            time_embedding: 时间嵌入，形状为 [batch_size, hidden_dim]。

        Returns:
            torch.Tensor: 更新后的 Token，形状与 tokens 相同。

        实现流程是将时间嵌入映射为两个分支的调制参数，分别执行注意力
        和 MLP 计算，并采用门控残差连接输出。
        """
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.condition(time_embedding).chunk(6, dim=-1)
        )
        normalized = self.norm1(tokens)
        normalized = normalized * (1.0 + scale_attention[:, None, :]) + shift_attention[:, None, :]
        attention_output, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + gate_attention[:, None, :] * attention_output

        normalized = self.norm2(tokens)
        normalized = normalized * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        tokens = tokens + gate_mlp[:, None, :] * self.mlp(normalized)
        return tokens


class VelocityField(nn.Module):
    """使用 DiT 预测叠加导频条件下的数据终点。

    Args:
        hidden_dim: Token 隐藏维度。
        depth: Transformer Block 数量。
        num_heads: 多头自注意力头数。
        patch_size: 时频 Patch 大小，格式为 (时间大小, 频率大小)。
        position_encoding: 位置编码样式，1d 使用展平序号，2d 使用时频坐标。

    Returns:
        VelocityField: 输入 [批次, 6, 时间, 频率]，输出 [批次, 2, 时间, 频率] 的终点预测。

    实现流程是将六通道时频图切分为 Patch，映射成 Token 并加入所选位置
    嵌入，再由多个 DiT Block 建模全局关系，最后通过线性投影恢复为数据
    实部和虚部的终点预测图。
    """

    def __init__(
        self,
        hidden_dim: int = 40,
        depth: int = 6,
        num_heads: int = 4,
        patch_size: tuple[int, int] = (2, 4),
        position_encoding: str = "1d",
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or depth < 1 or num_heads < 1:
            raise ValueError("hidden_dim、depth 和 num_heads 必须为正数")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        patch_time, patch_frequency = patch_size
        if patch_time < 1 or patch_frequency < 1:
            raise ValueError("patch_size 的两个维度必须为正数")
        if position_encoding not in {"1d", "2d"}:
            raise ValueError("position_encoding 必须是 1d 或 2d")

        self.patch_size = patch_size
        self.position_encoding = position_encoding
        self.patch_embedding = nn.Conv2d(
            6,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        time_fourier_dim = 40
        self.time_embedding = nn.Sequential(
            FourierTimeEmbedding(time_fourier_dim),
            nn.Linear(2 * time_fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.output_projection = nn.Linear(
            hidden_dim,
            2 * patch_time * patch_frequency,
        )
        # 输出前的 AdaLN 调制投影和最终输出投影使用零初始化。
        nn.init.zeros_(self.final_condition[-1].weight)
        nn.init.zeros_(self.final_condition[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, model_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """根据六通道条件和 Flow Matching 时间预测数据速度。

        Args:
            model_input: 输入特征，形状为 [batch_size, 6, time, frequency]。
            t: Flow Matching 时间，形状为 [batch_size, 1]，范围为 [0, 1]。

        Returns:
            torch.Tensor: 数据终点预测，形状为 [batch_size, 2, time, frequency]。

        实现流程是 Patch 化输入、加入固定正弦余弦位置编码，
        使用时间条件调制 Transformer Token，最后将 Patch 投影并还原到原始时频布局。
        """
        if model_input.ndim != 4 or model_input.shape[1] != 6:
            raise ValueError("model_input 的形状必须是 [batch_size, 6, time, frequency]")
        if t.ndim != 2 or t.shape[0] != model_input.shape[0] or t.shape[1] != 1:
            raise ValueError("t 的形状必须是 [batch_size, 1]")
        patch_time, patch_frequency = self.patch_size
        height, width = model_input.shape[-2:]
        if height % patch_time != 0 or width % patch_frequency != 0:
            raise ValueError("时间和频率维度必须能被 patch_size 整除")

        patch_features = self.patch_embedding(model_input)
        patch_height, patch_width = patch_features.shape[-2:]
        tokens = patch_features.flatten(2).transpose(1, 2)
        if self.position_encoding == "2d":
            position = self.position_embedding_2d(
                patch_height, patch_width, tokens.device, tokens.dtype
            )
        else:
            position = self.position_embedding(
                patch_height * patch_width, tokens.device, tokens.dtype
            )
        tokens = tokens + position
        time_embedding = self.time_embedding(t)
        for block in self.blocks:
            tokens = block(tokens, time_embedding)
        shift, scale = self.final_condition(time_embedding).chunk(2, dim=-1)
        normalized = self.final_norm(tokens)
        normalized = normalized * (1.0 + scale[:, None, :]) + shift[:, None, :]
        patches = self.output_projection(normalized)
        patches = patches.view(
            model_input.shape[0],
            patch_height,
            patch_width,
            2,
            patch_time,
            patch_frequency,
        )
        output = patches.permute(0, 3, 1, 4, 2, 5)
        return output.reshape(model_input.shape[0], 2, height, width)

    def position_embedding(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成标准一维正弦余弦位置嵌入。

        Args:
            count: Patch Token 数量。
            device: 目标设备。
            dtype: 目标数据类型。

        Returns:
            torch.Tensor: 形状为 [1, count, hidden_dim] 的位置嵌入。

        实现流程遵循《Attention Is All You Need》中的位置编码公式：偶数
        维使用正弦函数，奇数维使用余弦函数，频率由 10000 的幂次决定。
        Token 按展平后的时频 Patch 顺序使用一维位置索引。
        """
        hidden_dim = self.patch_embedding.out_channels
        positions = torch.arange(count, device=device, dtype=dtype).unsqueeze(1)
        frequencies = torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
        frequencies = torch.exp(-math.log(10000.0) * frequencies / hidden_dim)
        embedding = torch.zeros(count, hidden_dim, device=device, dtype=dtype)
        embedding[:, 0::2] = torch.sin(positions * frequencies)
        embedding[:, 1::2] = torch.cos(positions * frequencies[: embedding[:, 1::2].shape[1]])
        return embedding.unsqueeze(0)

    def position_embedding_2d(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成与时频 Patch 网格对应的二维正弦余弦位置嵌入。

        Args:
            height: 时间方向的 Patch 数量。
            width: 频率方向的 Patch 数量。
            device: 目标设备。
            dtype: 目标数据类型。

        Returns:
            torch.Tensor: 形状为 [1, height * width, hidden_dim] 的位置嵌入。

        实现流程是将隐藏维度分给时间轴和频率轴，分别按各轴坐标计算
        正弦余弦编码，再广播到二维网格并拼接。隐藏维度为奇数时，频率轴
        多分配一维。最后按时间在外、频率在内的顺序展平，与 Patch Token 一致。
        """
        hidden_dim = self.patch_embedding.out_channels
        time_dim = hidden_dim // 2
        axis_embeddings = []
        for count, dim in ((height, time_dim), (width, hidden_dim - time_dim)):
            positions = torch.arange(count, device=device, dtype=dtype).unsqueeze(1)
            frequencies = torch.arange(0, dim, 2, device=device, dtype=dtype)
            frequencies = torch.exp(-math.log(10000.0) * frequencies / dim)
            embedding = torch.zeros(count, dim, device=device, dtype=dtype)
            embedding[:, 0::2] = torch.sin(positions * frequencies)
            embedding[:, 1::2] = torch.cos(positions * frequencies[: dim // 2])
            axis_embeddings.append(embedding)
        time_embedding, frequency_embedding = axis_embeddings
        embedding = torch.cat(
            (
                time_embedding[:, None, :].expand(height, width, -1),
                frequency_embedding[None, :, :].expand(height, width, -1),
            ),
            dim=-1,
        )
        return embedding.reshape(1, height * width, hidden_dim)


def flow_matching_loss(
    model: nn.Module,
    channel: torch.Tensor,
    pilot: torch.Tensor,
    pilot_power_factor: float = 0.1,
    snr_db: float = 30.0,
) -> torch.Tensor:
    """计算三种调制混合数据的 Flow Matching 速度 MSE 损失。

    Args:
        model: 接收六通道条件并预测数据终点的 DiT。
        channel: 复数 SISO 信道，形状为 [batch_size, time, subcarrier]。
        pilot: 固定复数 QPSK 导频，形状为 [time, subcarrier]。
        pilot_power_factor: 导频功率因子，固定使用 0.1。
        snr_db: 接收信号信噪比，固定使用 30 dB。

    Returns:
        torch.Tensor: 预测数据速度与真实数据速度之间的标量 MSE。

    实现流程是每个样本等概率选择一种调制并生成数据，再按叠加导频公式构造发射信号，
    经过信道并加入 30 dB 复高斯噪声得到接收信号；随后在初始噪声
    和数据之间构造 Flow Matching 插值，时间满足 logit(t) ~ N(-0.8, 0.8²)。
    模型输出终点预测，再根据 v_theta = (x_theta - z_t) / max(1-t, 0.05)
    转换为速度并计算速度 MSE；损失包含 NaN 或 Inf 时立即报错。
    """
    if channel.ndim != 3 or not channel.is_complex():
        raise ValueError("channel 的形状必须是 [batch_size, time, subcarrier] 复数张量")
    if pilot.ndim != 2 or not pilot.is_complex():
        raise ValueError("pilot 的形状必须是 [time, subcarrier] 复数张量")
    if not 0.0 < pilot_power_factor < 1.0 or snr_db <= 0.0:
        raise ValueError("pilot_power_factor 必须在 0 和 1 之间，snr_db 必须为正数")

    batch_size, time_steps, subcarriers = channel.shape
    if pilot.shape != (time_steps, subcarriers):
        raise ValueError("pilot 的时间和子载波维度必须与 channel 一致")

    data = generate_mixed_data_signal(
        (batch_size, time_steps, subcarriers),
    ).to(channel.device)
    pilot = pilot.to(channel.device)
    received = generate_received_signal(
        channel,
        data,
        pilot,
        pilot_power_factor,
        snr_db,
    )

    data_real = torch.view_as_real(data).permute(0, 3, 1, 2)
    received_real = torch.view_as_real(received).permute(0, 3, 1, 2)
    pilot_real = torch.view_as_real(pilot).permute(2, 0, 1).unsqueeze(0)
    pilot_real = pilot_real.expand(batch_size, -1, -1, -1)

    x_0 = torch.randn_like(data_real)
    # 使用 logit-normal 分布采样时间：logit(t) ~ N(-0.8, 0.8²)。
    t = torch.sigmoid(-0.8 + 0.8 * torch.randn(batch_size, 1, device=channel.device))
    t_for_path = t[:, :, None, None]
    x_t = (1.0 - t_for_path) * x_0 + t_for_path * data_real
    model_input = torch.cat((x_t, received_real, pilot_real), dim=1)
    true_velocity = data_real - x_0
    endpoint_prediction = model(model_input, t)
    predicted_velocity = endpoint_to_velocity(endpoint_prediction, x_t, t)
    loss = nn.functional.mse_loss(predicted_velocity, true_velocity)
    if not torch.isfinite(loss):
        raise FloatingPointError("Flow Matching 损失包含 NaN 或 Inf")
    return loss


def endpoint_to_velocity(
    endpoint_prediction: torch.Tensor,
    interpolated_state: torch.Tensor,
    time: torch.Tensor,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """将终点预测按照 Flow Matching 公式转换为速度预测。

    Args:
        endpoint_prediction: 模型输出的终点预测 x_theta，形状为
            [batch_size, 2, time, frequency]。
        interpolated_state: 当前插值状态 z_t，形状与 endpoint_prediction 相同。
        time: 当前 Flow Matching 时间，形状为 [batch_size, 1]。
        epsilon: 分母下限，默认为 0.05，训练与推理使用相同截断。

    Returns:
        torch.Tensor: 转换后的速度预测，形状与 endpoint_prediction 相同。

    实现流程是将终点预测与当前插值状态作差，再除以 1-t，得到
    v_theta = (x_theta - z_t) / (1-t)。当 t 接近 1 时对分母进行下限
    截断，以避免数值不稳定。
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")
    if endpoint_prediction.shape != interpolated_state.shape:
        raise ValueError("终点预测和插值状态的形状必须一致")
    if time.ndim != 2 or time.shape[0] != endpoint_prediction.shape[0] or time.shape[1] != 1:
        raise ValueError("time 的形状必须是 [batch_size, 1]")
    denominator = (1.0 - time[:, :, None, None]).clamp_min(epsilon)
    return (endpoint_prediction - interpolated_state) / denominator


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    pilot: torch.Tensor,
    run: Any,
    global_step: int,
    max_grad_norm: float | None = None,
) -> tuple[float, int]:
    """训练模型一个 epoch。

    Args:
        model: 待训练的终点预测模型，输出通过公式转换为速度。
        loader: 提供目标分布样本的数据加载器。
        optimizer: 模型参数优化器。
        scheduler: 学习率调度器；plateau 模式在此仅推进 warmup。
        device: 训练设备。
        pilot: 固定复数 QPSK 导频，形状为 [时间, 子载波]。
        run: SwanLab 运行对象，用于记录 step 级指标。
        global_step: 当前 epoch 开始前的全局训练 step。
        max_grad_norm: 全模型梯度的 L2 范数上限；None 表示不启用梯度裁剪。

    Returns:
        tuple[float, int]: 当前 epoch 平均损失和更新后的全局 step。

    实现流程是遍历信道样本，按样本等概率混合三种调制，计算随机时间路径上的速度匹配损失，
    然后执行反向传播，按需裁剪梯度并记录裁剪后的范数，最后更新参数。
    """
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max-grad-norm 必须为有限正数")
    model.train()
    total_loss = 0.0
    total_samples = 0
    for channel in loader:
        channel = channel.to(device)
        loss = flow_matching_loss(model, channel, pilot)
        optimizer.zero_grad()
        loss.backward()
        current_learning_rate = optimizer.param_groups[0]["lr"]
        gradient_norm = calculate_gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("梯度范数包含 NaN 或 Inf，已停止参数更新")
        step_metrics = {
            "train/step_loss": loss.item(),
            "step/learning_rate": current_learning_rate,
            "step/gradient_norm": gradient_norm,
        }
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm, norm_type=2.0, error_if_nonfinite=True
            )
            step_metrics["step/clipped_gradient_norm"] = calculate_gradient_norm(model)
        optimizer.step()
        if isinstance(scheduler, WarmupReduceLROnPlateau):
            scheduler.step_batch()
        else:
            scheduler.step()
        global_step += 1
        run.log(step_metrics, step=global_step)
        total_loss += loss.item() * channel.shape[0]
        total_samples += channel.shape[0]
    return total_loss / total_samples, global_step


def calculate_gradient_norm(model: nn.Module) -> float:
    """计算模型所有参数梯度的 L2 范数。

    Args:
        model: 已完成反向传播、且参数中包含梯度的模型。

    Returns:
        float: 全模型梯度的 L2 范数。

    实现流程是遍历所有存在梯度的参数，累加梯度元素的平方和，最后开平方
    得到全模型梯度的 L2 范数。该指标仅用于记录，不会修改梯度或模型参数。
    """
    gradient_squared_norm = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            gradient_squared_norm += grad.square().sum()
    return torch.sqrt(gradient_squared_norm).item()


class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """结合 step 级线性 warmup 和验证损失驱动的学习率下降。

    Args:
        optimizer: 需要调整学习率的优化器。
        warmup_steps: 线性 warmup 的优化器更新次数，0 表示禁用。
        min_learning_rate: warmup 结束后的学习率下限。
        factor: 验证损失停滞时的学习率乘数。
        patience: 允许验证损失连续未改善的轮数，超过后降低学习率。

    Returns:
        WarmupReduceLROnPlateau: 支持保存和恢复完整状态的调度器。

    实现流程是通过 step_batch 在每次参数更新后推进 warmup；完成后停止
    按 batch 修改学习率，由调用方在后续每轮验证后调用 step(val_loss)。
    warmup 进度和初始学习率随父类 state_dict 一并保存。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        min_learning_rate: float,
        factor: float = 0.1,
        patience: int = 10,
    ) -> None:
        super().__init__(
            optimizer, mode="min", factor=factor, patience=patience,
            min_lr=min_learning_rate,
        )
        self.warmup_steps = warmup_steps
        self.completed_steps = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        if warmup_steps > 0:
            for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr / warmup_steps
            self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def step_batch(self) -> None:
        self.completed_steps += 1
        if self.completed_steps <= self.warmup_steps:
            scale = min((self.completed_steps + 1) / self.warmup_steps, 1.0)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * scale
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def build_step_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float,
    lr_scheduler: str = "cosine",
    plateau_factor: float = 0.1,
    plateau_patience: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler:
    """构造带线性 warmup 的余弦或 ReduceLROnPlateau 调度器。

    Args:
        optimizer: 需要调整学习率的优化器。
        total_steps: 整个训练过程的优化器更新次数。
        warmup_steps: 线性 warmup 的优化器更新次数。
        min_learning_rate: warmup 结束后的最小学习率。
        lr_scheduler: 学习率模式，cosine 或 plateau。
        plateau_factor: plateau 模式的学习率下降倍率。
        plateau_patience: plateau 模式允许验证损失未改善的轮数。

    Returns:
        torch.optim.lr_scheduler.LRScheduler: 所选模式对应的学习率调度器。

    实现流程是先将学习率从较小值线性增加到初始学习率，随后按照总训练
    step 数进行余弦衰减，或在后续每轮验证后根据损失停滞情况降低学习率。
    """
    if total_steps < 1:
        raise ValueError("total_steps 必须为正数")
    if warmup_steps < 0:
        raise ValueError("warmup_steps 不能为负数")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps 必须小于 total_steps，以保留学习率调整阶段")
    if lr_scheduler not in {"cosine", "plateau"}:
        raise ValueError("lr_scheduler 必须为 cosine 或 plateau")
    if not math.isfinite(plateau_factor) or not 0 < plateau_factor < 1:
        raise ValueError("plateau_factor 必须在 0 和 1 之间")
    if plateau_patience < 0:
        raise ValueError("plateau_patience 不能为负数")
    if lr_scheduler == "plateau":
        return WarmupReduceLROnPlateau(
            optimizer, warmup_steps, min_learning_rate,
            factor=plateau_factor, patience=plateau_patience,
        )
    base_learning_rate = optimizer.param_groups[0]["lr"]
    min_learning_rate_ratio = min_learning_rate / base_learning_rate
    decay_steps = max(total_steps - warmup_steps, 1)

    def learning_rate_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_lambda)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pilot: torch.Tensor,
    seed: int = 42,
) -> float:
    """在验证集上计算一个 epoch 的平均 Flow Matching 损失。

    Args:
        model: 已训练的终点预测模型。
        loader: 提供验证集信道样本的数据加载器。
        device: 评估设备。
        pilot: 固定复数 QPSK 导频，形状为 [时间, 子载波]。
        seed: 验证专用随机种子，默认值为 42。

    Returns:
        float: 验证集上的平均 Flow Matching 损失。

    实现流程是保存 CPU 和所有 CUDA 设备的随机状态，设置验证专用种子，
    然后使用与训练阶段相同的随机路径损失进行评估，不更新模型参数。
    固定数据加载顺序和批大小时，每次验证使用相同的样本调制选择、数据、噪声和时间采样。
    退出随机状态上下文时（包括发生异常时）恢复原状态，不影响后续训练。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for channel in loader:
            channel = channel.to(device)
            loss = flow_matching_loss(model, channel, pilot)
            total_loss += loss.item() * channel.shape[0]
            total_samples += channel.shape[0]
    return total_loss / total_samples


@torch.no_grad()
def recover_data_from_received(
    model: nn.Module,
    received: torch.Tensor,
    pilot: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """通过 Flow Matching 终点预测从接收信号恢复数据符号。

    Args:
        model: 已训练的 DiT 终点预测模型。
        received: 复数接收信号，形状为 [batch_size, time, frequency]。
        pilot: 固定复数 QPSK 导频，形状为 [time, frequency]。
        num_steps: Euler 积分步数。
        device: 推理设备。

    Returns:
        torch.Tensor: 恢复的数据实虚部，形状为 [batch_size, 2, time, frequency]。

    实现流程是从标准高斯噪声开始，将接收信号和导频转换为实虚部通道，
    再将 DiT 的终点预测转换为速度并进行 Euler 积分，得到最终的数据估计。
    """
    if received.ndim != 3 or not received.is_complex():
        raise ValueError("received 必须是 [batch_size, time, frequency] 复数张量")
    if num_steps < 1:
        raise ValueError("num_steps 必须是正整数")
    model.eval()
    batch_size, time_steps, subcarriers = received.shape
    received = received.to(device)
    pilot = pilot.to(device)
    received_real = torch.view_as_real(received).permute(0, 3, 1, 2)
    pilot_real = torch.view_as_real(pilot).permute(2, 0, 1).unsqueeze(0)
    pilot_real = pilot_real.expand(batch_size, -1, -1, -1)
    estimate = torch.randn(batch_size, 2, time_steps, subcarriers, device=device)
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step / num_steps, device=device)
        model_input = torch.cat((estimate, received_real, pilot_real), dim=1)
        endpoint_prediction = model(model_input, t)
        velocity = endpoint_to_velocity(endpoint_prediction, estimate, t)
        estimate = estimate + step_size * velocity
        if not torch.isfinite(estimate).all():
            raise FloatingPointError(f"Euler 积分第 {step + 1} 步的结果包含 NaN 或 Inf")
    return estimate


@torch.no_grad()
def evaluate_ber_vs_snr(
    model: nn.Module,
    loader: DataLoader,
    pilot: torch.Tensor,
    snr_values: list[float],
    flow_steps: int,
    device: torch.device,
    modulation: str = "QPSK",
    throughput_only: bool = False,
) -> list[dict[str, float | int]]:
    """在测试集上统计指定调制的 BER 和实测固定块吞吐量。

    Args:
        model: 已加载验证集最优权重的 DiT 模型。
        loader: 测试集信道数据加载器。
        pilot: 固定复数 QPSK 导频。
        snr_values: 待评估的 SNR 列表，单位为 dB。
        flow_steps: 数据恢复时使用的 Euler 积分步数。
        device: 推理设备。
        modulation: 数据调制方式，导频始终为 QPSK。
        throughput_only: 为 True 时只保存实测块吞吐量，不保存 BER。

    Returns:
        list[dict[str, float | int]]: 每个 SNR 对应的实测块指标；
            非仅吞吐量模式还包含 BER、错误比特数和总比特数。

    实现流程是为每个 SNR 按指定调制方式生成数据和接收信号，使用 Flow
    Matching 采样恢复数据，再按最近星座点判决与 Gray 编码统计比特错误率。
    将每帧数据位连续分成 128 位块，直接统计无错块并换算吞吐量。
    """
    if not snr_values:
        raise ValueError("snr_values 不能为空")
    results = []
    model.eval()
    snr_progress = tqdm(snr_values, desc=f"测试 {modulation} SNR", unit="snr")
    for snr_db in snr_progress:
        bit_errors = 0
        total_bits = 0
        successful_blocks = total_blocks = total_ofdm_symbols = 0
        snr_progress.set_postfix(snr_db=f"{snr_db:g} dB")
        for channel in loader:
            channel = channel.to(device)
            batch_size, time_steps, subcarriers = channel.shape
            data = generate_data_signal(
                (batch_size, time_steps, subcarriers),
                modulation=modulation,
            ).to(device)
            received = generate_received_signal(channel, data, pilot, 0.1, snr_db)
            estimate = recover_data_from_received(
                model,
                received,
                pilot,
                flow_steps,
                device,
            )
            target_bits = demodulate_bits(data, modulation)
            predicted_bits = demodulate_bits(
                torch.complex(estimate[:, 0], estimate[:, 1]), modulation
            )
            good, blocks = count_successful_blocks(target_bits, predicted_bits)
            successful_blocks += good
            total_blocks += blocks
            total_ofdm_symbols += batch_size * time_steps
            if not throughput_only:
                bit_errors += int((predicted_bits != target_bits).sum().item())
                total_bits += int(target_bits.numel())
        record = {"snr_db": float(snr_db), **block_throughput_metrics(
            successful_blocks, total_blocks, total_ofdm_symbols
        )}
        if not throughput_only:
            record.update(ber=bit_errors / total_bits, bit_errors=bit_errors,
                          total_bits=total_bits)
        results.append(record)
    return results


def evaluate_all_modulations(
    model: nn.Module,
    loader: DataLoader,
    pilot: torch.Tensor,
    snr_values: list[float],
    flow_steps: int,
    device: torch.device,
    throughput_only: bool = False,
) -> dict[str, list[dict[str, float | int]]]:
    """使用同一个模型分别测试 QPSK、16QAM 和 64QAM。

    Args:
        model: 已加载权重的 DiT 模型，不接收调制类型条件。
        loader: 可重复遍历的测试集信道数据加载器。
        pilot: 三种调制共用的固定复数 QPSK 导频。
        snr_values: 每种调制均需测试的 SNR 列表，单位为 dB。
        flow_steps: 数据恢复时使用的 Euler 积分步数。
        device: 推理设备。
        throughput_only: 为 True 时只保存实测块吞吐量。

    Returns:
        dict[str, list[dict[str, float | int]]]: 以调制名称为键，每个值为
            各 SNR 的吞吐量记录；非仅吞吐量模式还包含 BER。

    实现流程是依次遍历三种调制，在完整测试集和所有指定 SNR 上分别评估。
    各自采用对应星座的 Gray 比特判决，独立统计结果。
    """
    return {
        modulation: evaluate_ber_vs_snr(
            model, loader, pilot, snr_values, flow_steps, device, modulation,
            throughput_only=throughput_only,
        )
        for modulation in MODULATIONS
    }


def save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    save_dir: Path,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
) -> None:
    """保存当前验证集最优模型及其训练配置。

    Args:
        model: 待保存的模型。
        optimizer: 当前优化器，用于保存训练状态。
        scheduler: 当前学习率调度器，用于保存调度状态。
        save_dir: 模型和配置文件的保存目录。
        epoch: 当前最优模型对应的 epoch 编号。
        val_loss: 当前最优验证损失。
        config: 需要写入 JSON 的训练配置字典。

    Returns:
        None: 在 save_dir 下保存 fmreceiver.pt 和 fmreceiver.json。

    实现流程是先创建保存目录，再保存模型、优化器和调度器状态，最后
    将训练参数、数据形状和最优指标转换为 JSON 写入配置文件。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, save_dir / "fmreceiver.pt")

    config_to_save = dict(config)
    config_to_save.update(
        {
            "best_epoch": epoch,
            "best_val_loss": val_loss,
            "checkpoint_path": str(save_dir / "fmreceiver.pt"),
        }
    )
    with (save_dir / "fmreceiver.json").open("w", encoding="utf-8") as config_file:
        json.dump(config_to_save, config_file, ensure_ascii=False, indent=2)


def find_dataset_paths(
    data_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Path]:
    """根据文件后缀识别目录中的数据集。

    按所需划分扫描目录直属文件，每种后缀必须且只能匹配一个文件，
    避免缺失数据集或在多个候选文件之间误选。

    Args:
        data_dir: 数据集所在目录，支持用户主目录符号 ~。
        splits: 需要识别的数据划分，默认包含训练集、验证集和测试集。

    Returns:
        dict[str, Path]: 数据划分名称到对应 .pt 文件路径的映射。

    Raises:
        NotADirectoryError: 指定路径不是有效目录。
        FileNotFoundError: 某个数据划分没有对应文件。
        ValueError: 某个数据划分匹配到多个文件。
    """
    data_dir = data_dir.expanduser()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据集目录不存在或不是目录: {data_dir}")
    paths = {}
    for split in splits:
        pattern = f"*_{split}.pt"
        matches = sorted(path for path in data_dir.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"数据集目录 {data_dir} 中找不到 {pattern} 文件")
        if len(matches) > 1:
            filenames = ", ".join(path.name for path in matches)
            raise ValueError(
                f"数据集目录 {data_dir} 中存在多个 {pattern} 文件: {filenames}。"
                "请指定每种后缀仅包含一个文件的目录。"
            )
        paths[split] = matches[0]
    return paths


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

        Returns:
            argparse.ArgumentParser: 支持数据路径、训练轮数、批大小和随机种子的解析器。

    该函数将最小实验配置集中到命令行，方便快速调整 Flow Matching 实验。
    """
    parser = argparse.ArgumentParser(
        description="基于 CDL 信道的 Flow Matching 接收机：混合训练并分别测试 QPSK、16QAM、64QAM"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="运行模式：训练后测试，或加载指定权重仅测试",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="仅测试模式使用的模型权重文件路径",
    )
    parser.add_argument("--throughput-only", action="store_true",
                        help="仅测试模式只统计实测 128 位数据块吞吐量")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/cdl-c/normalized"),
        help="数据集目录，自动识别 *_train.pt、*_val.pt、*_test.pt 文件",
    )
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument(
        "--position-encoding",
        choices=["1d", "2d"],
        default="1d",
        help="位置编码样式：一维展平序号（默认）或二维时频坐标；测试时须与训练一致",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="每轮固定训练的 step 数，使用有放回随机采样；默认遍历完整训练集",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="初始学习率")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="启用全模型 L2 梯度范数裁剪并指定范数上限（有限正数）；默认不启用",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="warmup 结束后的最小学习率")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="线性 warmup 的训练轮数")
    parser.add_argument(
        "--lr-scheduler", choices=["cosine", "plateau"], default="cosine",
        help="warmup 后的学习率模式：余弦衰减（默认）或 ReduceLROnPlateau",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.1, help="plateau 学习率下降倍率")
    parser.add_argument("--plateau-patience", type=int, default=10, help="plateau 允许验证损失未改善的轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--snr-values",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help="BER 评估使用的 SNR 列表，单位为 dB",
    )
    parser.add_argument("--flow-steps", type=int, default=5, help="测试时的 Flow Matching Euler 步数")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs"),
        help="最优模型和训练配置的保存目录，文件名固定为 fmreceiver",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="模型版本，格式为 YYYYmmdd_vn，例如 20260920_v1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在版本目录中的模型、配置和测试结果",
    )
    parser.add_argument("--swanlab-project", type=str, default="Research-Flow_Matching_Receiver_Final", help="SwanLab 项目名称")
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="SwanLab 实验名称，默认与本次运行的版本号一致",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="disabled",
        help="SwanLab 运行模式",
    )
    parser.add_argument(
        "--swanlab-api-key",
        type=str,
        default=None,
        help="SwanLab API Key；也可以通过 SWANLAB_API_KEY 环境变量提供",
    )
    return parser


def main() -> None:
    """运行三种调制混合训练、验证及逐调制测试流程。

    Returns:
        None: 按命令行配置保存最优权重、训练配置及三种调制的测试结果。

    实现流程是初始化信道数据集和终点预测模型，在训练和验证时按样本等概率
    混合三种调制，使用固定 QPSK 导频和 30 dB 噪声构造条件输入。训练结束后
    加载最优权重，分别测试三种调制；仅测试模式直接加载指定权重执行相同评估。
    """
    args = build_parser().parse_args()
    if args.mode == "test" and args.checkpoint is None:
        raise ValueError("仅测试模式必须通过 --checkpoint 指定模型权重文件")
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("--checkpoint 仅能与 --mode test 一起使用")
    if args.throughput_only and args.mode != "test":
        raise ValueError("--throughput-only 仅能与 --mode test 一起使用")
    if args.version is None:
        args.version = f"{datetime.now().strftime('%Y%m%d')}_v1"
    if args.swanlab_name is None:
        args.swanlab_name = args.version
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须是正整数")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("steps-per-epoch 必须是正整数")
    if args.max_grad_norm is not None and (
        not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
    ):
        raise ValueError("max-grad-norm 必须为有限正数")
    if args.flow_steps < 1:
        raise ValueError("flow-steps 必须是正整数")
    if not all(math.isfinite(value) for value in args.snr_values):
        raise ValueError("snr-values 必须全部为有限数值")
    if not math.isfinite(args.learning_rate) or not math.isfinite(args.min_learning_rate):
        raise ValueError("learning-rate 和 min-learning-rate 必须为有限数值")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("learning-rate 必须为正数，min-learning-rate 不能为负数")
    if args.min_learning_rate >= args.learning_rate:
        raise ValueError("min-learning-rate 必须小于 learning-rate")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs 不能为负数")
    if not math.isfinite(args.plateau_factor) or not 0 < args.plateau_factor < 1:
        raise ValueError("plateau-factor 必须在 0 和 1 之间")
    if args.plateau_patience < 0:
        raise ValueError("plateau-patience 不能为负数")
    if args.mode == "train" and args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs 必须小于 epochs；短实验请显式减小 warmup-epochs")
    if re.fullmatch(r"\d{8}_v\d+", args.version) is None:
        raise ValueError("version 格式必须是 YYYYmmdd_vn，例如 20260920_v1")

    splits = ("test",) if args.mode == "test" else ("train", "val", "test")
    dataset_paths = find_dataset_paths(args.data_dir, splits)
    for split, path in dataset_paths.items():
        setattr(args, f"{split}_path", path)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = ChannelDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = VelocityField(position_encoding=args.position_encoding).to(device)
    pilot = generate_fixed_pilot(
        test_dataset.channel.shape[1],
        test_dataset.channel.shape[2],
        seed=args.seed,
    ).to(device)

    if args.mode == "test":
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError("模型权重文件缺少 model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])
        output_dir = args.save_dir / args.version
        output_dir.mkdir(parents=True, exist_ok=True)
        test_results = evaluate_all_modulations(
            model=model,
            loader=test_loader,
            pilot=pilot,
            snr_values=args.snr_values,
            flow_steps=args.flow_steps,
            device=device,
            throughput_only=args.throughput_only,
        )
        test_metrics = {
            "version": args.version,
            "checkpoint_path": str(checkpoint_path),
            "test_path": str(args.test_path),
            "best_epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("val_loss"),
            "flow_steps": args.flow_steps,
            "modulations": list(MODULATIONS),
            "position_encoding": args.position_encoding,
            "snr_values": args.snr_values,
            "results": test_results,
            "metric_type": "measured_uncoded_128_bit_block_throughput" if args.throughput_only else "ber_and_measured_throughput",
        }
        result_name = "throughput_metrics.json" if args.throughput_only else "test_metrics.json"
        result_path = output_dir / result_name
        version = 2
        while result_path.exists():
            result_path = output_dir / f"{Path(result_name).stem}_v{version}.json"
            version += 1
        with result_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(test_metrics, metrics_file, ensure_ascii=False, indent=2)
        print(f"测试指标已保存至 {result_path}")
        return

    train_dataset = ChannelDataset(args.train_path)
    val_dataset = ChannelDataset(args.val_path)
    channel_shape = train_dataset.channel.shape[1:]
    if val_dataset.channel.shape[1:] != channel_shape:
        raise ValueError("训练集和验证集的信道特征维度不一致")
    if test_dataset.channel.shape[1:] != channel_shape:
        raise ValueError("训练集和测试集的信道特征维度不一致")
    if args.steps_per_epoch is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        # 有放回采样保证每轮恰好训练指定 step 数，且每个 batch 均为指定大小。
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=args.steps_per_epoch * args.batch_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler
        )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = build_step_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_learning_rate=args.min_learning_rate,
        lr_scheduler=args.lr_scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
    )

    version_save_dir = args.save_dir / args.version
    if version_save_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出路径已存在: {version_save_dir}。"
            "如需覆盖，请添加 --overwrite 参数。"
        )

    swanlab_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"swanlab_api_key"}
    }
    swanlab_config.update(
        modulations=list(MODULATIONS),
        modulation_sampling="per_sample_uniform",
        modulation_conditioning=False,
    )
    api_key = args.swanlab_api_key or os.getenv("SWANLAB_API_KEY")
    if api_key and args.swanlab_mode != "disabled":
        swanlab.login(api_key=api_key, save=False)
    run = swanlab.init(
        log_dir=str(version_save_dir / "swanlog"),
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=None,
        config=swanlab_config,
    )

    save_config = {
        **swanlab_config,
        "device": str(device),
        "model": {
            "name": "DiT",
            "input_shape": [6, train_dataset.channel.shape[1], train_dataset.channel.shape[2]],
            "output_shape": [2, train_dataset.channel.shape[1], train_dataset.channel.shape[2]],
            "hidden_dim": 40,
            "depth": 6,
            "num_heads": 4,
            "patch_size": [2, 4],
            "position_encoding": args.position_encoding,
        },
        "pilot_power_factor": 0.1,
        "snr_db": 30.0,
        "time_sampling": {"distribution": "logit_normal", "mean": -0.8, "std": 0.8},
        "velocity_denominator_min": 0.05,
    }

    try:
        progress_bar = tqdm(range(1, args.epochs + 1), desc="训练进度", unit="epoch")
        global_step = 0
        best_val_loss = float("inf")
        for epoch in progress_bar:
            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                pilot,
                run,
                global_step,
                max_grad_norm=args.max_grad_norm,
            )
            val_loss = evaluate_one_epoch(
                model, val_loader, device, pilot
            )
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(f"第 {epoch} 轮的训练或验证损失包含 NaN 或 Inf")
            if isinstance(scheduler, WarmupReduceLROnPlateau) and epoch > args.warmup_epochs:
                scheduler.step(val_loss)
            current_learning_rate = optimizer.param_groups[0]["lr"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_dir=version_save_dir,
                    epoch=epoch,
                    val_loss=val_loss,
                    config={
                        **save_config,
                        "learning_rate_after_scheduler": current_learning_rate,
                    },
                )
            run.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                },
                step=epoch,
            )
            progress_bar.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
                lr=f"{current_learning_rate:.2e}",
            )
    except Exception as error:
        run.finish(state="crashed", error=str(error))
        raise
    else:
        run.finish()

    best_checkpoint_path = version_save_dir / "fmreceiver.pt"
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_results = evaluate_all_modulations(
        model=model,
        loader=test_loader,
        pilot=pilot,
        snr_values=args.snr_values,
        flow_steps=args.flow_steps,
        device=device,
    )
    test_metrics = {
        "version": args.version,
        "test_path": str(args.test_path),
        "best_epoch": best_checkpoint["epoch"],
        "best_val_loss": best_checkpoint["val_loss"],
        "flow_steps": args.flow_steps,
        "modulations": list(MODULATIONS),
        "position_encoding": args.position_encoding,
        "snr_values": args.snr_values,
        "results": test_results,
        "metric_type": "ber_and_measured_throughput",
    }
    result_path = version_save_dir / "test_metrics.json"
    version = 2
    while result_path.exists():
        result_path = version_save_dir / f"test_metrics_v{version}.json"
        version += 1
    with result_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(test_metrics, metrics_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
