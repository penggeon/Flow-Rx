"""加载 Flow-Rx 权重并对外部接收网格进行硬判决。"""

from __future__ import annotations

from pathlib import Path

import torch

from .flow_rx import (
    VelocityField,
    demodulate_bits,
    generate_fixed_pilot,
    recover_data_from_received,
)


def load_receiver(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> VelocityField:
    """从可信的检查点文件载入论文采用的终点预测模型。

    Args:
        checkpoint_path: 包含 ``model_state_dict`` 的 PyTorch 检查点路径。
        device: 权重及后续推理使用的设备。

    Returns:
        VelocityField: 处于评估模式的六层 DiT 接收机。

    实现流程是初始化与论文一致的网络，仅读取检查点中的张量和基础类型，
    恢复模型参数，然后迁移到目标设备并切换为评估模式。
    """
    target = torch.device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    model = VelocityField(position_encoding="1d")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(target).eval()


@torch.no_grad()
def detect_bits(
    model: VelocityField,
    received: torch.Tensor,
    pilot: torch.Tensor,
    modulation: str,
    num_steps: int = 5,
    generator_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从已完成同步和 FFT 的复数接收网格恢复数据与比特。

    Args:
        model: 已载入权重的 Flow-Rx 模型。
        received: 形状为 ``[B, 14, 64]`` 的 complex64 接收网格。
        pilot: 形状为 ``[14, 64]`` 的已知复数 QPSK 导频。
        modulation: ``QPSK``、``16QAM`` 或 ``64QAM``。
        num_steps: Euler 积分次数，论文默认使用 5 次。
        generator_seed: 若提供，设置一次采样随机种子以便复现结果。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 复数数据估计 ``[B,14,64]``
            和布尔比特网格 ``[B,14,64,bits_per_symbol]``。

    实现流程是在标准高斯初态上运行条件 Euler 采样，将实虚部组合成复数
    数据估计，最后按原始实现的 QAM Gray 标记执行最近邻硬判决。
    """
    if received.ndim != 3 or not received.is_complex() or received.shape[-2:] != (14, 64):
        raise ValueError("received 必须是 [B,14,64] 的复数张量")
    if pilot.shape != (14, 64) or not pilot.is_complex():
        raise ValueError("pilot 必须是 [14,64] 的复数张量")
    if generator_seed is not None:
        torch.manual_seed(generator_seed)
    device = next(model.parameters()).device
    estimate = recover_data_from_received(model, received, pilot, num_steps, device)
    symbols = torch.complex(estimate[:, 0], estimate[:, 1])
    return symbols, demodulate_bits(symbols, modulation)


def paper_pilot(seed: int = 123) -> torch.Tensor:
    """生成与附带 seed123 权重配套的固定 QPSK 导频。

    Args:
        seed: 导频随机种子，附带权重使用 123。

    Returns:
        torch.Tensor: 形状为 ``[14,64]`` 的复数导频网格。
    """
    return generate_fixed_pilot(14, 64, seed)
