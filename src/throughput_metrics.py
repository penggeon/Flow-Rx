"""统一统计固定长度未编码数据块的实测吞吐量。"""

from __future__ import annotations

import torch


BLOCK_SIZE_BITS = 128


def count_successful_blocks(
    target_bits: torch.Tensor,
    predicted_bits: torch.Tensor,
    block_size_bits: int = BLOCK_SIZE_BITS,
) -> tuple[int, int]:
    """从真实判决比特统计完全正确的数据块。

    Args:
        target_bits: [B,...]，首维为独立帧，其余维为数据位。
        predicted_bits: 与 target_bits 同形状的判决比特。
        block_size_bits: 每个未编码数据块的固定比特数。

    Returns:
        tuple[int, int]: 完全正确的数据块数和总数据块数。

    每帧按时频顺序展平数据比特，连续切成等长块；正交导频
    应在调用前去除导频 RE。任一比特出错即判该块错误。
    """
    if target_bits.shape != predicted_bits.shape or target_bits.ndim < 2:
        raise ValueError("目标比特和判决比特必须具有相同的 [B,...] 形状")
    if block_size_bits < 1:
        raise ValueError("block_size_bits 必须为正整数")
    target = target_bits.reshape(target_bits.shape[0], -1).bool()
    predicted = predicted_bits.reshape(predicted_bits.shape[0], -1).bool()
    bits_per_frame = target.shape[1]
    if bits_per_frame == 0 or bits_per_frame % block_size_bits:
        raise ValueError(
            f"每帧数据位数 {bits_per_frame} 必须是固定块长 {block_size_bits} 的整数倍"
        )
    correct = (target == predicted).reshape(-1, block_size_bits).all(dim=1)
    return int(correct.sum().item()), int(correct.numel())


def block_throughput_metrics(
    successful_blocks: int,
    total_blocks: int,
    total_ofdm_symbols: int,
    block_size_bits: int = BLOCK_SIZE_BITS,
) -> dict[str, float | int]:
    """将实测数据块成功数换算为 BLER 和平均吞吐量。

    Args:
        successful_blocks: 完全正确的数据块数。
        total_blocks: 实际传输的数据块总数。
        total_ofdm_symbols: 测试过程中占用的 OFDM 符号总数。
        block_size_bits: 与分块统计一致的固定比特数。

    Returns:
        dict: 块长、成功块数、总块数、错误块数、实测 BLER，以及
            平均 bit/OFDM 符号。吞吐量已包含专用导频 RE 开销。
    """
    if not 0 <= successful_blocks <= total_blocks or total_blocks < 1:
        raise ValueError("数据块数量无效")
    if total_ofdm_symbols < 1:
        raise ValueError("OFDM 符号总数必须为正")
    return {
        "block_size_bits": block_size_bits,
        "successful_blocks": successful_blocks,
        "total_blocks": total_blocks,
        "block_errors": total_blocks - successful_blocks,
        "total_ofdm_symbols": total_ofdm_symbols,
        "bler": (total_blocks - successful_blocks) / total_blocks,
        "throughput": block_size_bits * successful_blocks / total_ofdm_symbols,
    }
