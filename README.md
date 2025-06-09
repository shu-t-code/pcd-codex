# Point Cloud Edge Classification

This repository contains example code for classifying edge points in 3D point clouds using a simple graph-based convolutional neural network.

## Dataset Preparation

The training and validation data are provided as `.npy` files:

- **Point clouds**: arrays of shape `(N, 6)` containing `[x, y, z, nx, ny, nz]`.
- **Labels**: arrays of shape `(N, 1)` with `0` for non-edge and `1` for edge.

You can store the data anywhere on your system. Create text files listing the paths to the point and label pairs:

```text
/path/to/cloud1.npy /path/to/labels1.npy
/path/to/cloud2.npy /path/to/labels2.npy
...
```

Pass the text file to the training script using `--train-list` (and `--val-list` for validation). Relative paths are supported, so you might place your files in a `data/` directory at the project root and reference them from the lists.

## Training

Install PyTorch, PyTorch Geometric and scikit-learn, then run:

```bash
python train.py --train-list train_list.txt --val-list val_list.txt --epochs 10
```

Use `--amp` to enable mixed precision training.
