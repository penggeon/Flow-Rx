"""生成符合 3GPP TR 38.901 的 CDL 复数时频信道矩阵。

依赖：
    sionna==2.1.0
    torch

返回的频域信道矩阵计算公式为：

    H(f) = sum_l a_l exp(-j*2*pi*f*tau_l),

其中 ``a_l`` 和 ``tau_l`` 由 Sionna 的 3GPP CDL 模型生成。
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from importlib.metadata import version
from pathlib import Path

import torch

from sionna.phy import config as sionna_config
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.channel.tr38901 import CDL, PanelArray


def channel_sampling_frequency(args: argparse.Namespace) -> float:
    if args.sampling_frequency is not None:
        return args.sampling_frequency
    return 14 * args.subcarrier_spacing / 15


def create_cdl_model(args: argparse.Namespace, device: str | None = None) -> CDL:
    """构造可在多个批次间复用的 CDL 模型。

    Args:
        args: 信道、天线及运动参数。
        device: 计算设备；为空时自动选择 CUDA 或 CPU。

    Returns:
        CDL: 按指定方向连接基站与终端阵列的信道模型。

    先构造单极化 SISO 或可配置 MIMO 阵列，再设置姿态、速度和标准版本。
    """
    # Sionna 2.1.0 要求 CUDA 设备包含设备编号。
    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # SISO 两端均使用一个单极化天线单元；MIMO 保持可配置的双极化基站阵列。
    bs_rows = 1 if args.siso else args.bs_rows
    bs_cols = 1 if args.siso else args.bs_cols
    ut_rows = 1 if args.siso else args.ut_rows
    ut_cols = 1 if args.siso else args.ut_cols
    bs_array = PanelArray(
        num_rows_per_panel=bs_rows,
        num_cols_per_panel=bs_cols,
        polarization="single" if args.siso else "dual",
        polarization_type="V" if args.siso else "cross",
        antenna_pattern=args.bs_pattern,
        precision="single",
        carrier_frequency=args.carrier_frequency,
        device=selected_device,
    )
    # 用户终端阵列：ut_rows*ut_cols 个单极化垂直天线单元。
    ut_array = PanelArray(
        num_rows_per_panel=ut_rows,
        num_cols_per_panel=ut_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern=args.ut_pattern,
        precision="single",
        carrier_frequency=args.carrier_frequency,
        device=selected_device,
    )

    return CDL(
        model=args.model,
        delay_spread=args.delay_spread,
        carrier_frequency=args.carrier_frequency,
        ut_array=ut_array,
        bs_array=bs_array,
        direction=args.direction,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        spec_version="16.1",
        precision="single",
        bs_orientation=args.bs_orientation,
        ut_orientation=args.ut_orientation,
        ut_velocity=args.ut_velocity,
        device=selected_device,
    )


def generate_cdl_channel_matrix(
    args: argparse.Namespace,
    device: str | None = None,
    cdl: CDL | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成复数时频信道，保留收发天线维度。

    Args:
        args: 批次大小、时间点数、频率点数及信道配置。
        device: 新建模型时使用的设备。
        cdl: 可复用模型；为空时按 args 创建。

    Returns:
        tuple: H 为 [B,T,Rx,Tx,F]，时延为 [B,Path]（秒），
            路径系数为 [B,T,Rx,Tx,Path]。

    在均匀时刻生成路径系数，使用 Sionna 官方函数构造频率网格并转换为
    频域响应，最后调整维度。转换时不归一化，统一由训练集 RMS 缩放。
    默认时间间隔近似 NR 普通 CP 的平均符号间隔，并非精确符号边界。
    """
    if cdl is None:
        cdl = create_cdl_model(args, device)
    a, tau = cdl(
        batch_size=args.batch_size,
        num_time_steps=args.time_steps,
        sampling_frequency=channel_sampling_frequency(args),
    )

    # a[:,0,:,0,:,:,:] 的形状为 [批次, 接收端, 发射端, 路径, 时间]，
    # 调整为 [批次, 时间, 接收天线, 发射天线, 路径]。
    path_coefficients = a[:, 0, :, 0, :, :, :].permute(0, 4, 1, 2, 3)
    delays = tau[:, 0, 0, :]

    frequencies = subcarrier_frequencies(
        num_subcarriers=args.subcarriers,
        subcarrier_spacing=args.subcarrier_spacing,
        precision="double" if tau.dtype == torch.float64 else "single",
        device=str(tau.device),
    )
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
    # 官方输出为 [B,1,Rx,1,Tx,T,F]，仅移除收发节点维，保留天线维。
    channel_matrix = h_freq[:, 0, :, 0, :, :, :].permute(0, 3, 1, 2, 4)

    return channel_matrix, delays, path_coefficients


def build_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """根据生成参数构造输出路径。

    Args:
        args: 输出目录和信道配置。

    Returns:
        tuple: 复数矩阵与 JSON 配置的路径。

    创建输出目录，将模型、样本数量、时频及天线参数拼接为文件名。
    """
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    antenna_suffix = (
        "_SISO"
        if args.siso
        else f"_BS{args.bs_rows}x{args.bs_cols}_UT{args.ut_rows}x{args.ut_cols}"
    )

    file_stem = (
        f"cdl_{args.model}"
        f"_{args.direction}"
        f"_N{args.total_samples}"
        f"_B{args.batch_size}"
        f"_T{args.time_steps}"
        f"_F{args.subcarriers}"
        f"_SCS{args.subcarrier_spacing:g}"
        f"_Fc{args.carrier_frequency / 1e9:g}GHz"
        f"_DS{args.delay_spread * 1e9:g}ns"
        f"{antenna_suffix}"
        f"_V{args.min_speed:g}-{args.max_speed:g}mps"
    )
    return output_dir / f"{file_stem}.pt", output_dir / f"{file_stem}.json"


def validate_arguments(args: argparse.Namespace) -> None:
    """在生成前检查数值、模型适用范围和切分结果。

    Args:
        args: 命令行参数。

    Returns:
        None: 参数有效时正常返回，否则抛出 ValueError。

    先检查有限值与正值，再检查速度、采样率及数据集是否存在空集合。
    """
    for name in ("total_samples", "batch_size", "time_steps", "subcarriers"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} 必须是正整数")
    if not args.siso:
        for name in ("bs_rows", "bs_cols", "ut_rows", "ut_cols"):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} 必须是正整数")
    for name in ("subcarrier_spacing", "carrier_frequency", "delay_spread"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    if args.sampling_frequency is not None and (
        not math.isfinite(args.sampling_frequency) or args.sampling_frequency <= 0
    ):
        raise ValueError("sampling_frequency 必须为有限正数")
    if not 0.5e9 <= args.carrier_frequency <= 100e9:
        raise ValueError("TR 38.901 载频适用范围为 0.5—100 GHz")
    if args.subcarriers * args.subcarrier_spacing > 2e9:
        raise ValueError("当前模型频率窗口带宽不能超过 2 GHz")
    for name in ("min_speed", "max_speed"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须为有限非负数")
    if args.max_speed < args.min_speed:
        raise ValueError("max_speed 不能小于 min_speed")
    for name in ("bs_orientation", "ut_orientation", "ut_velocity"):
        vector = getattr(args, name)
        if vector is not None and any(not math.isfinite(x) for x in vector):
            raise ValueError(f"{name} 的分量必须为有限数")
    if args.seed is not None and not 0 <= args.seed < 2**64 - 1:
        raise ValueError("seed 必须在 [0, 2**64-1) 内")
    speed = math.hypot(*args.ut_velocity) if args.ut_velocity is not None else args.max_speed
    doppler = speed * args.carrier_frequency / 299792458.0
    if channel_sampling_frequency(args) <= 2 * doppler:
        warnings.warn("时间采样率不高于两倍最大 Doppler，时间序列可能混叠", stacklevel=2)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(not math.isfinite(r) or not 0 < r < 1 for r in ratios):
        raise ValueError("划分比例必须是 0 到 1 之间的有限数")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("划分比例之和必须为 1")
    train = int(args.total_samples * args.train_ratio)
    val = int(args.total_samples * args.val_ratio)
    if min(train, val, args.total_samples - train - val) < 1:
        raise ValueError("切分后存在空集合，请增加 total_samples 或调整划分比例")


def generate_dataset(args: argparse.Namespace) -> torch.Tensor:
    """复用模型分批生成复数信道并写入预分配的 CPU 张量。

    Args:
        args: 总样本数、批次大小和模型参数。

    Returns:
        torch.Tensor: 形状为 [N,T,Rx,Tx,F] 的 CPU 复数张量。

    每批生成独立信道实现，逐批复制到最终数组，避免拼接产生整份副本。
    """
    cdl = create_cdl_model(args)
    channel_matrices = None
    generated_samples = 0

    while generated_samples < args.total_samples:
        current_batch_size = min(
            args.batch_size,
            args.total_samples - generated_samples,
        )
        round_args = argparse.Namespace(**vars(args))
        round_args.batch_size = current_batch_size

        H, _, _ = generate_cdl_channel_matrix(round_args, cdl=cdl)
        if not torch.isfinite(H).all():
            raise ValueError("生成的信道包含 NaN 或 Inf")
        if args.siso and H.shape[2:4] != (1, 1):
            raise ValueError("SISO 模式的收发天线维必须为 1")
        if channel_matrices is None:
            channel_matrices = torch.empty((args.total_samples, *H.shape[1:]), dtype=H.dtype)
        channel_matrices[generated_samples:generated_samples + current_batch_size] = H.cpu()
        generated_samples += current_batch_size

    return channel_matrices


def split_dataset(
    H: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """使用独立随机数生成器打乱并切分数据集。

    Args:
        H: 完整复数信道张量，首维为样本数。
        args: 切分比例与随机种子。

    Returns:
        tuple: 训练、验证、测试张量及三者样本数量。

    先生成独立的随机排列，再按比例向下取整，余数归入测试集。"""
    # 在切分前随机打乱样本，避免训练集、验证集和测试集存在顺序偏差。
    generator = torch.Generator(device=H.device)
    if args.seed is None:
        generator.seed()
    else:
        generator.manual_seed(args.seed)
    permutation = torch.randperm(H.shape[0], device=H.device, generator=generator)
    H = H[permutation]

    train_size = int(H.shape[0] * args.train_ratio)
    val_size = int(H.shape[0] * args.val_ratio)
    test_size = H.shape[0] - train_size - val_size
    if min(train_size, val_size, test_size) < 1:
        raise ValueError("切分后的训练、验证及测试集均须非空")

    train_end = train_size
    val_end = train_size + val_size
    train_H = H[:train_end]
    val_H = H[train_end:val_end]
    test_H = H[val_end:]

    return train_H, val_H, test_H, (train_size, val_size, test_size)


def normalize_datasets(
    train_H: torch.Tensor,
    val_H: torch.Tensor,
    test_H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """使用训练集的全局 RMS 同时归一化三个数据集。

    Args:
        train_H: 训练集复数矩阵。
        val_H: 验证集复数矩阵。
        test_H: 测试集复数矩阵。

    Returns:
        tuple: 归一化后的三个矩阵和用于恢复原始幅度的 RMS。

    仅用训练集统计平均功率，检查 RMS 有效后对三个集合使用同一缩放。"""
    if train_H.shape[0] == 0:
        raise ValueError("训练集不能为空，否则无法计算训练集 RMS")

    # 只使用训练集计算一个全局 RMS，验证集和测试集不参与统计量计算。
    train_rms = torch.sqrt(torch.mean(torch.abs(train_H) ** 2))
    train_rms_value = float(train_rms.item())
    if not math.isfinite(train_rms_value) or train_rms_value <= 0.0:
        raise ValueError("训练集 RMS 必须为有限正数")

    train_H = train_H / train_rms
    val_H = val_H / train_rms
    test_H = test_H / train_rms

    return train_H, val_H, test_H, train_rms_value


def save_datasets(
    train_H: torch.Tensor,
    val_H: torch.Tensor,
    test_H: torch.Tensor,
    args: argparse.Namespace,
    sample_sizes: tuple[int, int, int],
    train_rms: float,
    *,
    normalized: bool = True,
    output_paths: tuple[Path, Path] | None = None,
) -> None:
    """保存三个数据集文件和一个 JSON 配置文件。

    Args:
        train_H: 待保存的训练集。
        val_H: 待保存的验证集。
        test_H: 待保存的测试集。
        args: 生成及输出配置。
        sample_sizes: 三个集合的样本数量。
        train_rms: 训练集归一化前的全局 RMS。
        normalized: 输入张量是否已经按训练集 RMS 归一化。
        output_paths: 共用的矩阵与配置基础路径，用于配对两份数据的文件名。

    Returns:
        None: 将复数张量及实验配置写入磁盘。

    根据归一化状态选择子目录，以排他创建保存张量，再记录缩放状态与坐标。
    """
    train_size, val_size, test_size = sample_sizes
    matrix_path, config_path = output_paths or build_output_paths(args)
    output_dir = matrix_path.parent / ("normalized" if normalized else "raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / matrix_path.name
    config_path = output_dir / config_path.name
    train_path = matrix_path.with_name(f"{matrix_path.stem}_train.pt")
    val_path = matrix_path.with_name(f"{matrix_path.stem}_val.pt")
    test_path = matrix_path.with_name(f"{matrix_path.stem}_test.pt")

    for path, tensor in ((train_path, train_H), (val_path, val_H), (test_path, test_H)):
        with path.open("xb") as file:
            # 切分结果是共享底层存储的视图，复制以免每个文件保存整个数据集。
            torch.save(tensor.clone(), file)

    config = {
        "model": args.model,
        "direction": args.direction,
        "total_samples": args.total_samples,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "subcarriers": args.subcarriers,
        "subcarrier_spacing_hz": args.subcarrier_spacing,
        "sampling_frequency_hz": (
            args.sampling_frequency
            if args.sampling_frequency is not None
            else 14 * args.subcarrier_spacing / 15
        ),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "normalization": {
            "method": "global_rms" if normalized else "none",
            "source": "train" if normalized else None,
            "train_rms": train_rms,
        },
        "shuffled_before_split": True,
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
        "carrier_frequency_hz": args.carrier_frequency,
        "delay_spread_s": args.delay_spread,
        "siso": args.siso,
        "bs_rows": None if args.siso else args.bs_rows,
        "bs_cols": None if args.siso else args.bs_cols,
        "ut_rows": None if args.siso else args.ut_rows,
        "ut_cols": None if args.siso else args.ut_cols,
        "min_speed_mps": args.min_speed,
        "max_speed_mps": args.max_speed,
        "train_shape": list(train_H.shape),
        "val_shape": list(val_H.shape),
        "test_shape": list(test_H.shape),
        "matrix_dtype": str(train_H.dtype),
        "train_file": train_path.name,
        "val_file": val_path.name,
        "test_file": test_path.name,
    }
    config.update(channel_metadata(args))
    with config_path.open("x", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"训练集已保存：{train_path}")
    print(f"验证集已保存：{val_path}")
    print(f"测试集已保存：{test_path}")
    print(f"配置文件已保存：{config_path}")
    print(f"训练集 H 的形状：{tuple(train_H.shape)}")
    print(f"验证集 H 的形状：{tuple(val_H.shape)}")
    print(f"测试集 H 的形状：{tuple(test_H.shape)}")


def channel_metadata(args: argparse.Namespace) -> dict:
    """记录复数矩阵的坐标、模型约定和运行环境。

    Args:
        args: 已验证的生成参数。

    Returns:
        dict: 可直接写入 JSON 的配置和坐标。

    从有效采样率构造时间轴，以整数子载波索引构造频率轴，补充模型设置。
    """
    sampling = channel_sampling_frequency(args)
    return {
        "schema_version": 2,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "spec_version": "16.1",
        "sionna_version": version("sionna"),
        "torch_version": torch.__version__,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "precision": "single",
        "axis_order": ["sample", "time", "rx_antenna", "tx_antenna", "frequency"],
        "time_sampling": "uniform",
        "time_reference": "每个独立信道实现从 t=0 开始，不表示精确 NR 符号边界",
        "sampling_frequency_hz": sampling,
        "time_s": [t / sampling for t in range(args.time_steps)],
        "frequency_grid": "centered_fft",
        "frequency_reference_hz": args.carrier_frequency,
        "frequency_offset_hz": subcarrier_frequencies(
            args.subcarriers, args.subcarrier_spacing, precision="single", device="cpu"
        ).tolist(),
        "bs_pattern": args.bs_pattern,
        "ut_pattern": args.ut_pattern,
        "bs_polarization": "V" if args.siso else "cross",
        "ut_polarization": "V",
        "bs_orientation_rad": args.bs_orientation,
        "ut_orientation_rad": args.ut_orientation,
        "ut_velocity_mps": args.ut_velocity,
        "velocity_distribution": "fixed" if args.ut_velocity is not None else
            "Sionna 随机三维方向，速率在 min_speed 到 max_speed 均匀分布",
    }


def parse_arguments() -> argparse.Namespace:
    """解析信道生成命令行参数。

    Returns:
        argparse.Namespace: 信道、天线、数据切分和输出配置。

    按通用与天线参数注册选项，默认生成 SISO 数据集。"""
    parser = argparse.ArgumentParser(description="生成符合 3GPP 标准的 CDL 信道矩阵")
    common_group = parser.add_argument_group("通用参数")
    common_group.add_argument(
        "--output",
        type=Path,
        default=Path("datasets"),
        help="输出根目录；raw 保存原始数据，normalized 保存归一化数据",
    )
    common_group.add_argument(
        "--model",
        type=str.upper,
        choices=["A", "B", "C", "D", "E"],
        default="C",
        help="CDL 模型类型，可选 A、B、C、D、E，默认使用 C",
    )
    common_group.add_argument(
        "--direction",
        choices=["uplink", "downlink"],
        default="uplink",
        help="链路方向，可选 uplink 或 downlink，默认使用 uplink",
    )
    common_group.add_argument("--total-samples", type=int, default=10000)
    common_group.add_argument("--batch-size", type=int, default=100)
    common_group.add_argument("--time-steps", type=int, default=14)
    common_group.add_argument("--subcarriers", type=int, default=64)
    common_group.add_argument("--subcarrier-spacing", type=float, default=15e3)
    common_group.add_argument(
        "--sampling-frequency",
        type=float,
        default=None,
        help="信道时变采样率（Hz）；默认使用 NR 普通 CP 平均符号速率的均匀采样近似",
    )
    common_group.add_argument("--carrier-frequency", type=float, default=3.5e9)
    common_group.add_argument("--delay-spread", type=float, default=300e-9)
    common_group.add_argument("--min-speed", type=float, default=0.0)
    common_group.add_argument("--max-speed", type=float, default=35.0)
    common_group.add_argument("--seed", type=int, default=42)
    common_group.add_argument("--train-ratio", type=float, default=0.8)
    common_group.add_argument("--val-ratio", type=float, default=0.1)
    common_group.add_argument("--test-ratio", type=float, default=0.1)
    siso_group = parser.add_argument_group("SISO 参数")
    siso_group.add_argument(
        "--siso",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认生成 SISO，使用 --no-siso 启用 MIMO",
    )
    mimo_group = parser.add_argument_group("MIMO 天线参数（SISO 模式下忽略）")
    mimo_group.add_argument("--bs-rows", type=int, default=2)
    mimo_group.add_argument("--bs-cols", type=int, default=2)
    mimo_group.add_argument("--ut-rows", type=int, default=1)
    mimo_group.add_argument("--ut-cols", type=int, default=1)
    common_group.add_argument("--bs-pattern", choices=["38.901", "omni"], default="38.901")
    common_group.add_argument("--ut-pattern", choices=["38.901", "omni"], default="omni")
    common_group.add_argument("--bs-orientation", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                              help="基站姿态的三个欧拉角，单位为弧度")
    common_group.add_argument("--ut-orientation", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                              help="终端姿态的三个欧拉角，单位为弧度")
    common_group.add_argument("--ut-velocity", type=float, nargs=3, default=None,
                              help="固定三维速度 vx vy vz（m/s），设置后覆盖速度范围")
    return parser.parse_args()


def main() -> None:
    """执行复数信道生成流程。

    Returns:
        None: 保存复数矩阵及配置文件。

    验证参数后初始化随机流，再分批生成，使用独立随机排列切分，
    将原始切分保存到 raw，并按训练集 RMS 归一化后保存到 normalized。
    两份数据使用相同文件名和样本顺序。
    """
    args = parse_arguments()
    validate_arguments(args)

    # 只在程序开始时设置一次随机种子，避免不同批次重复生成相同随机 realization。
    if args.seed is not None:
        sionna_config.seed = args.seed
        torch.manual_seed(args.seed)

    H = generate_dataset(args)
    train_H, val_H, test_H, sample_sizes = split_dataset(H, args)
    normalized_train, normalized_val, normalized_test, train_rms = normalize_datasets(
        train_H,
        val_H,
        test_H,
    )
    output_paths = build_output_paths(args)
    save_datasets(
        train_H, val_H, test_H, args, sample_sizes, train_rms,
        normalized=False, output_paths=output_paths,
    )
    save_datasets(
        normalized_train, normalized_val, normalized_test, args, sample_sizes, train_rms,
        normalized=True, output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
