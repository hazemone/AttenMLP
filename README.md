# AttenMLP: An Attention-Enhanced Multilayer Perceptron for 3D Point Cloud Classification

Official implementation of the paper **"AttenMLP: An Attention-Enhanced Multilayer Perceptron for 3D Point Cloud Classification"** by Rohit Mohanraj (National Institute of Technology Warangal) and Hazem Eissa (University of Sunderland).

<!-- TODO: add the paper link once published, e.g. [Paper](https://...) -->

## Overview

AttenMLP is a lightweight architecture for 3D point cloud classification that combines the efficiency of MLP-based designs with the representational power of attention mechanisms. It addresses two persistent limitations of existing MLP approaches (e.g., PointMLP, DualMLP): limited capacity for long-range dependency modelling and incomplete preservation of spatial information across deep transformations.

The architecture is built on three core components:

1. **Dual-pathway architecture**: point features are processed simultaneously in a *sparse* stream (hierarchical FPS+KNN grouping capturing global geometric structure) and a *dense* stream (fine-grained local detail), fused by a parameter-free operator.
2. **Sinusoidal positional encoding**: spatial context derived from raw 3D coordinates is injected into the attention input, preserving spatial cohesion of points throughout deep network transformations.
3. **Efficient multi-head attention**: attention is *localised*: it operates on the FPS-downsampled centroid set (M ≪ N) within KNN neighbourhoods, so the quadratic cost applies to M rather than the full point count N, avoiding the O(N²) complexity of full transformers.

Evaluated on the ModelNet40 benchmark, AttenMLP achieves **92.4% mean class accuracy (mAcc)** and **93.85% overall accuracy (OA)** with only **3.15 MB of parameters** and **1.85 GFLOPs** — a notably lightweight footprint relative to transformer-based methods such as Point-MAE (15.2 MB, 5.12 GFLOPs) and PCT (9.14 MB, 3.85 GFLOPs).

## Results on ModelNet40

| Model | Type | Params (MB) | FLOPs (GB) | mAcc (%) | OA (%) |
| --- | --- | --- | --- | --- | --- |
| PointNet | Point | 3.47 | 0.45 | 86.0 | 89.2 |
| PointCNN | Point | 2.37 | 1.84 | 88.1 | 92.2 |
| PointNet++ | Point | 1.48 | 1.68 | — | 91.9 |
| DGCNN | Graph | 1.81 | 2.43 | 90.2 | 92.2 |
| LGEFE | Graph | 2.59 | 0.97 | 91.3 | 93.9 |
| CVA-GNN | Graph | — | — | 91.8 | 93.7 |
| Point-MAE | Transformer | 15.2 | 5.12 | 92.1 | 93.5 |
| PCT | Transformer | 9.14 | 3.85 | 91.8 | 93.5 |
| PointMLP | MLP | 3.95 | 2.15 | 91.6 | 93.9 |
| DualMLP | MLP | — | — | 91.8 | 93.8 |
| **AttenMLP (Ours)** | **MLP** | **3.15** | **1.85** | **92.4** | **93.85** |

### Ablation study

| Configuration | mAcc (%) | OA (%) | Params (MB) |
| --- | --- | --- | --- |
| Base MLP | 89.4 | 90.20 | 2.85 |
| + Multi-Head Attention | 91.8 | 92.70 | 3.02 |
| + Positional Encoding | 90.6 | 91.50 | 2.92 |
| **AttenMLP (Full)** | **92.4** | **93.85** | **3.15** |

Adding multi-head attention alone yields a +2.5% OA gain, while positional encoding alone adds +1.3%. Combining both components produces a synergistic interaction and the best overall accuracy. Eight attention heads were found to be the optimal accuracy–cost trade-off.

## Installation

```shell
# clone this repo
git clone https://github.com/hazemone/AttenMLP.git
cd AttenMLP
```

Requirements:

- Python ≥ 3.9
- PyTorch (with CUDA — training requires a GPU)
- Additional packages: `h5py`, `numpy`, `scikit-learn`, `tqdm`, `thop` (optional, for FLOPs/params profiling), `pointnet2_ops` (optional, for CUDA-accelerated farthest point sampling)

```shell
pip install h5py numpy scikit-learn tqdm thop
```

## Dataset

Experiments use the **ModelNet40** benchmark: 12,311 CAD models across 40 object categories (9,843 training / 2,468 test samples). Each model is represented by 1,024 points sampled from the object surface and normalised to fit within a unit sphere centred at the origin.

The dataset (`modelnet40_ply_hdf5_2048`) is downloaded automatically on the first run of `main.py` and placed under `classification_ModelNet40/data/`. If the automatic download fails, download `modelnet40_ply_hdf5_2048.zip` manually and extract it to `classification_ModelNet40/data/modelnet40_ply_hdf5_2048/`.

## Usage

### Training on ModelNet40

```shell
cd classification_ModelNet40

# train AttenMLP with the default (paper) configuration
python main.py --msg attention_dual_512-1024 --seed 0
```

By default, training creates a folder named `checkpoints/{modelName}-{msg}-{seed}` containing `args.txt`, `best_checkpoint.pth`, `last_checkpoint.pth`, `log.txt`, and `out.txt`. Training automatically resumes from `last_checkpoint.pth` if it exists in the checkpoint folder.

### Important parameters

```yaml
dual_net: True          # enable the dual-pathway (sparse + dense) model
num_points_low: 512     # number of input points for the sparse pathway
num_points_high: 1024   # number of input points for the dense pathway
neighbours_low: 32      # KNN neighbours for the sparse pathway
neighbours_high: 32     # KNN neighbours for the dense pathway
embed_dim: 28           # embedding dimension of the sparse pathway
num_channel: 28         # embedding dimension of the dense pathway
last_layer_concat: add  # fusion operator for the two pathways (add / concat)
batch_size: 16          # training batch size
epoch: 100              # number of training epochs
learning_rate: 0.025    # initial learning rate (cosine annealed)
min_lr: 0.005           # minimum learning rate
weight_decay: 2e-4      # SGD weight decay
seed: 0                 # random seed (deterministic mode enabled)
```

### Training configuration (paper setup)

| Parameter | Value |
| --- | --- |
| Batch size | 16 |
| Epochs | 100 |
| Input points | 1,024 |
| Learning rate (init.) | 0.025 |
| Minimum learning rate | 0.005 |
| LR scheduler | Cosine annealing |
| Optimizer | SGD (momentum 0.9) |
| Weight decay | 2 × 10⁻⁴ |

Training uses mixed precision (FP16) and was conducted on an NVIDIA RTX 3070 GPU. Runs are seeded and deterministic algorithms are enabled for reproducibility.

## Repository structure

```
AttenMLP/
├── classification_ModelNet40/
│   ├── main.py            # training / evaluation entry point
│   ├── data.py            # ModelNet40 dataset loading, download, and augmentation
│   ├── helper.py          # loss and utility functions
│   ├── models/
│   │   └── pointmlp.py    # AttenMLP model: dual pathways, multi-head attention,
│   │                      # sinusoidal positional encoding, FPS+KNN grouping
│   └── utils/             # logging, checkpointing, progress utilities
├── LICENSE
└── README.md
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{mohanraj2026attenmlp,
  title   = {AttenMLP: An Attention-Enhanced Multilayer Perceptron for 3D Point Cloud Classification},
  author  = {Mohanraj, Rohit and Eissa, Hazem},
  year    = {2026}
}
```

<!-- TODO: update the BibTeX entry with the venue, volume, and pages once published -->

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgment

Our implementation is built upon the following excellent codebases. We thank the authors for their great work, which made our implementation possible: [PointMLP](https://github.com/ma-xu/pointMLP-pytorch), [DualMLP](https://github.com/snehaputul/DualMLP)

## Contact

For questions about the paper or the code, please open an issue or contact:

- Hazem Eissa — hazem.eissa@sunderland.ac.uk ([ORCID](https://orcid.org/0000-0002-3645-1247))
- Rohit Mohanraj — rm22ecb0f08@student.nitw.ac.in ([ORCID](https://orcid.org/0009-0002-1574-8305))
