import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from typing import Literal
import sklearn.utils

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

print("Keras version:", keras.__version__)


def get_running_environment() -> (
    Literal["local", "colab", "kaggle", "marimo", "jupyter"]
):
    """
    Returns the current running environment.
    """
    if "COLAB_GPU" in os.environ:
        return "colab"
    elif "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    elif "MARIMO" in os.environ:
        return "marimo"
    elif "JUPYTERHUB_USER" in os.environ:
        return "jupyter"
    else:
        return "local"


def get_data_path(running_env: str) -> str:
    """
    Returns the path to the data directory based on the running environment.
    """
    if running_env == "colab":
        return "/content/data"
    elif running_env == "kaggle":
        return "/kaggle/input/histopathologic-cancer-detection"
    elif running_env == "marimo":
        return "/data"
    else:
        return "./data/histopathologic-cancer-detection"


def set_random_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to {seed}")


def main():
    current_env = get_running_environment()
    print(f"Running environment: {current_env}")
    DATA_DIR = get_data_path(current_env)
    print(f"Data path: {DATA_DIR} (for {current_env} environment)")

    RANDOM_SEED = 42
    IMAGE_SIZE = 96
    IMAGE_CHANNEL = 3
    SAMPLE_SIZE = 1000
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train")
    TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test")

    set_random_seed(RANDOM_SEED)

    df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    negative_samples = df[df["label"] == 0].sample(
        SAMPLE_SIZE, random_state=RANDOM_SEED
    )
    positive_samples = df[df["label"] == 1].sample(
        SAMPLE_SIZE, random_state=RANDOM_SEED
    )

    train_df = sklearn.utils.shuffle(
        pd.concat([negative_samples, positive_samples], ignore_index=True)
    )

    # read dataset