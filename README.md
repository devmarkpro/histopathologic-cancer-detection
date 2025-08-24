# Histopathologic Cancer Detection

Deep learning models to classify histopathologic image patches as cancerous or non-cancerous for the Kaggle competition.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Data Loading](#data-loading)
  - [Training](#training)
  - [Inference & Submission](#inference--submission)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains code and resources for the Histopathologic Cancer Detection Kaggle competition. Participants build and evaluate convolutional neural networks to identify tumor tissue in 96×96 image patches. A positive label indicates that at least one pixel in the central 32×32 region is tumor.

## Project Structure

```
├── main.ipynb              # Notebook for EDA, modeling, and submission
├── image_dataset.py        # PyTorch Dataset for train/test image patches
├── data/                   # Dataset files
│   ├── train/              # Training TIFF images
│   ├── test/               # Test TIFF images
│   └── train_labels.csv    # CSV of ground-truth labels
├── checkpoints/            # Model weights (e.g., SimpleCNN, ResNet50)
├── submissions/            # Generated submission CSVs
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions
└── README.md               # This file
```

## Requirements

- Python 3.13 or later
- See `pyproject.toml` and `uv.lock` for pinned dependency versions:
  - torch, torchvision
  - numpy, pandas, scikit-image, scikit-learn
  - matplotlib, tqdm, ipykernel

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/histopathologic-cancer-detection.git
cd histopathologic-cancer-detection

# Install package and dependencies
pip install .

# (Optional) Install GraphViz to visualize model architectures
brew install graphviz
```

## Dataset

1. Register and download data from Kaggle: https://www.kaggle.com/c/histopathologic-cancer-detection/data
2. Unzip and place under `data/`:
   - `data/train/`
   - `data/test/`
   - `data/train_labels.csv`
   - (Optional) `data/sample_submission.csv`

## Usage

### Jupyter Notebook

Launch Jupyter and open the notebook:

```bash
jupyter lab  # or jupyter notebook
```
Follow cells in `main.ipynb` for data exploration, model training, and submission generation.

### Data Loading

Use the `ImageDataset` class for efficient loading:

```python
from image_dataset import ImageDataset
from torchvision import transforms
import pandas as pd

labels = pd.read_csv('data/train_labels.csv', index_col=0)
dataset = ImageDataset(data_dir='data', transform=transforms.ToTensor(), label_df=labels)
```

### Training

Customize model architectures and hyperparameters in the notebook. Supports CPU, GPU, and Apple MPS devices. Training checkpoints are saved to `checkpoints/`.

### Inference & Submission

After training, run the inference cells in `main.ipynb` to generate a submission CSV under `submissions/`. Submit to Kaggle to evaluate performance.
