# Skoltech Self-Driving Car Project

This project studies occupancy grid prediction from multi-camera vehicle images using PyTorch. It combines dataset loading, model training, validation, inference, and qualitative visualization in a compact research-style codebase.

The repository was developed as an academic computer vision / deep learning project focused on bird's-eye-view scene understanding from synchronized automotive cameras.

## Project Overview

The main goal is to predict a BEV occupancy map from four input camera views. The codebase includes:

- a baseline image-to-grid model;
- geometry-aware BEV models;
- training and validation pipelines;
- inference utilities for generating predicted occupancy grids;
- a notebook for experiments, visualizations, and quick comparisons.

## Repository Structure

- `main.py` - main entry point for training and inference;
- `conf/` - Hydra configuration files for models, losses, and runtime settings;
- `src/dataset.py` - dataset loader for images, camera calibration matrices, and occupancy targets;
- `src/train.py` - model training and validation loop;
- `src/inference.py` - inference pipeline for generating predicted occupancy grids;
- `src/models/` - model implementations;
- `submissions/` - saved checkpoints and generated predictions;
- `baseline_v4.ipynb` - exploratory notebook with baseline training, visualization, and model comparison.

## Task Definition

Each sample consists of four synchronized camera images together with camera parameters and a ground-truth occupancy grid. The task is to infer a top-down binary occupancy representation of the surrounding scene.

The current implementation uses:

- 4 camera images per sample;
- 4 intrinsic matrices;
- 4 `car_to_cam` extrinsic matrices;
- `gt_occupancy_grid` for supervised training and validation.

Input images are resized to `256x512` and normalized with ImageNet statistics.

## Dataset Layout

The dataset is expected to be stored outside the repository, next to this project directory.

By default, the paths in `conf/config.yaml` are:

- `../data/autonomy_yandex_dataset_train/`
- `../data/autonomy_yandex_dataset_val/`
- `../data/autonomy_yandex_dataset_test/`

Each split must contain an `info.csv` file. The CSV stores relative paths to image files, calibration matrices, and occupancy grid arrays.

Expected layout:

```text
Skoltech-Self-Driving-Car-project/
├── project/
└── data/
    ├── autonomy_yandex_dataset_train/
    │   └── info.csv
    ├── autonomy_yandex_dataset_val/
    │   └── info.csv
    └── autonomy_yandex_dataset_test/
        └── info.csv
```

The loader resolves sample files relative to the parent directory of each split, so the relative paths inside `info.csv` should remain unchanged.

## Implemented Models

The project includes three model families:

- `multi_cam_bev` - a simple multi-camera baseline with per-camera feature extraction, feature fusion, and BEV decoding;
- `lss_bev` - a Lift-Splat-Shoot style model that uses camera geometry;
- `bev_fusion_net` - a more advanced BEV fusion architecture with frustum-based view transformation.

The default configuration uses `multi_cam_bev`.

## Training

Training is managed with [Hydra](https://hydra.cc/).

Run the default training configuration:

```bash
python main.py
```

Explicit training mode:

```bash
python main.py mode=train
```

Example with a different model and loss:

```bash
python main.py mode=train model=lss_bev loss=bev_loss
```

Example with custom hyperparameters:

```bash
python main.py mode=train train.batch_size=8 train.epochs=10 train.num_workers=4
```

The best checkpoint is saved to:

```text
submissions/${model.name}/model.pt
```

The latest checkpoint is saved to:

```text
submissions/${model.name}/model_last.pt
```

## Inference

Run inference with the configured checkpoint:

```bash
python main.py mode=infer
```

Run inference with an explicit checkpoint:

```bash
python main.py mode=infer paths.infer_checkpoint=submissions/resnet50_bev/model.pt
```

Predictions are saved to:

```text
submissions/${model.name}/predicted_static_grids/
```

The corresponding `info.csv` file is copied into the same output directory.

## Notebook Experiments

The notebook `baseline_v4.ipynb` contains:

- dataset inspection;
- camera and occupancy-grid visualization;
- baseline model definition and quick training;
- prediction export;
- comparison between the simple baseline and a trained BEV model.

This notebook is intended for experimentation and qualitative analysis rather than as the main training interface.

## Configuration

The main runtime configuration is stored in `conf/config.yaml`.

It defines:

- execution mode;
- dataset paths;
- checkpoint paths;
- optimizer and scheduler settings;
- batch size, learning rate, number of epochs, and other training parameters.

Model configurations:

- `conf/model/multi_cam_bev.yaml`
- `conf/model/lss_bev.yaml`
- `conf/model/bev_fusion_net.yaml`

Loss configurations:

- `conf/loss/bce.yaml`
- `conf/loss/dice.yaml`
- `conf/loss/bev_loss.yaml`

## Evaluation

Validation is performed using IoU on the predicted occupancy grid, ignoring cells marked with value `255`.

## Notes

- The repository currently does not include a dependency lockfile such as `requirements.txt` or `pyproject.toml`.
- Some model backbones rely on pretrained torchvision weights.
- Geometry-aware models depend on correct intrinsic and extrinsic calibration data.
- A trained checkpoint may already be available in `submissions/`.

## Quick Start

1. Place the dataset in the expected `data/` directory next to the project folder.
2. Check the dataset paths in `conf/config.yaml`.
3. Start training:

```bash
python main.py mode=train
```

4. Or run inference:

```bash
python main.py mode=infer
```
