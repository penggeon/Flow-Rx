# Flow-Rx

Code accompanying the unpublished manuscript **FlowRx: A Conditional Flow-Matching OFDM Receiver for Superimposed-Pilot**.

Flow-Rx learns a conditional transport from Gaussian noise to a clean QAM data grid. A time-conditioned patch Transformer predicts the clean endpoint from the intermediate state, received grid, and known pilot. One model handles QPSK, 16QAM, and 64QAM. This repository contains the receiver implementation, CDL channel generation, BER and 128-bit block-throughput evaluation, and one trained CDL-C checkpoint.

## Contents

| Path | Purpose |
| --- | --- |
| `src/flow_rx.py` | Training, validation, five-step Euler inference, BER and throughput evaluation |
| `src/predict.py` | Python API for external received grids |
| `src/cdl_channel.py` | 3GPP CDL channel generation via Sionna |
| `src/throughput_metrics.py` | Successful 128-bit block count and measured throughput |
| `checkpoints/seed123/` | CDL-C mixed-modulation checkpoint and original training metadata |

Experiment variants, baselines, intermediate weights, paper build files, and generated datasets are outside this package. The checkpoint is about 2.7 MB; generated datasets stay under `datasets/` and are ignored by Git.

## Setup

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The channel generator was developed with Sionna 2.1.0. The recorded training environment used PyTorch 2.14.0+cu130. A CUDA device is helpful for the full experiment; the receiver chooses CPU when CUDA is unavailable. Install a PyTorch build compatible with your system before installing the remaining packages if necessary.

## Reproduce the channel dataset

```bash
python -m src.cdl_channel --output datasets/cdl-c --model C \
  --total-samples 10000 --batch-size 100 --seed 42 --siso
```

This generates 8,000/1,000/1,000 channel realizations for training/validation/testing and saves raw and train-RMS-normalized tensors. The receiver reads `datasets/cdl-c/normalized`. Frame shape is 14 OFDM symbols by 64 subcarriers; carrier frequency is 3.5 GHz, subcarrier spacing 15 kHz, delay spread 300 ns, and terminal speed is uniformly sampled from 0 to 35 m/s. Generation may take time and disk space. Data symbols, the fixed pilot, and AWGN are generated during training and evaluation.

## Train

```bash
python -m src.flow_rx --mode train \
  --data-dir datasets/cdl-c/normalized \
  --epochs 300 --steps-per-epoch 200 --batch-size 256 \
  --learning-rate 0.002 --warmup-epochs 50 --min-learning-rate 0.00001 \
  --flow-steps 5 --seed 123 --version 20261020_v2 \
  --swanlab-mode disabled
```

The script selects the checkpoint with the lowest validation loss and then evaluates all three modulations. Outputs go to `runs/<version>/`. The manuscript specifies pilot fraction 0.1, training SNR 30 dB, logit-normal flow time with mean -0.8 and standard deviation 0.8, six DiT blocks, hidden width 40, four attention heads, and 2×4 patches. The checkpoint in this package was trained with seed 123; its full original settings are in `checkpoints/seed123/training_config.json`. That file retains paths from the original experiment as provenance; use the commands above in this package.

## Evaluate the included checkpoint

After generating the CDL-C dataset:

```bash
python -m src.flow_rx --mode test \
  --checkpoint checkpoints/seed123/fmreceiver.pt \
  --data-dir datasets/cdl-c/normalized \
  --seed 123 --flow-steps 5 --version 20261020_v2 \
  --snr-values -10 -5 0 5 10 15 20
```

Results are written to `runs/20261020_v2/test_metrics.json`. Test mode requires only `*_test.pt`. The same model is evaluated separately for QPSK, 16QAM, and 64QAM; modulation is used for data generation and final hard demodulation, not as a network input. The random pilot depends on `--seed`, so use `123` with the included checkpoint. BER and throughput also vary slightly with evaluation randomness.

## Use received grids directly

```python
import torch

from src.predict import detect_bits, load_receiver, paper_pilot

model = load_receiver("checkpoints/seed123/fmreceiver.pt", device="cpu")
pilot = paper_pilot(seed=123)
received = torch.load("my_received_grid.pt", weights_only=True)  # complex [B, 14, 64]
symbols, bits = detect_bits(model, received, pilot, "16QAM", num_steps=5,
                            generator_seed=123)
```

`received` must be a synchronized, CP-removed, FFT-domain SISO grid with the same pilot, grid layout, and channel normalization convention as training. `symbols` has shape `[B,14,64]`; `bits` has shape `[B,14,64,log2(M)]`. Use a checkpoint only from a trusted source.

## Reproducibility note

The final experiment code uses Gray labeling for 16QAM and 64QAM (`src/flow_rx.py:demodulate_bits`). The manuscript describes the same labeling. QPSK uses sign-based bit decisions.

The package implements uncoded SISO reception under ideal timing and carrier synchronization. It does not include a deployed radio front end or MIMO receiver.

## Citation

The manuscript has not been published. If you refer to this work, cite it as: Zhenpeng Ge, Yitong Liu, Meng Zou, and Hongwen Yang, “FlowRx: A Conditional Flow-Matching OFDM Receiver for Superimposed-Pilot,” unpublished manuscript, 2026. Publication details will be added when available.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
