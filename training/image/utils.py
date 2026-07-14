"""
=========================================================
FilterX V2 - Utility Functions
=========================================================

Shared utilities used across training, evaluation
and model conversion.

Responsibilities:
- Logging
- JSON read/write
- Training history persistence
- Model summary export
- Timing utilities
- Random seed setup
"""

import json
import logging
import pickle
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
from config import DPI
import numpy as np
import tensorflow as tf
from config import MODEL_SUMMARY_PATH

# Logger
def setup_logger(name: str = "FilterX") -> logging.Logger:
    """
    Creates and returns a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
logger = setup_logger()

# Random Seed
def set_seed(seed: int) -> None:
    # Set all random seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# JSON Utilities
def save_json(data: dict, filepath: Path) -> None:
    # Save dictionary as JSON.
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: Path) -> dict:
    # Load JSON file.
    with open(filepath, "r") as f:
        return json.load(f)

# Pickle Utilities
def save_pickle(obj, filepath: Path) -> None:
    # Save object using pickle.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path):
    # Load pickle object.
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Model Summary
def save_model_summary(model, filepath= MODEL_SUMMARY_PATH) -> None:
    # Save model.summary() into a text file.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

# Timing Utility
class Timer:
    """
    Simple timer class.

    Example
    -------
    timer = Timer()

    timer.start()

    ...

    elapsed = timer.stop()
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        return time.perf_counter() - self.start_time

# Directory Utility
def ensure_directory(path: Path) -> None:
    # Creates directory if it does not exist.
    path.mkdir(parents=True, exist_ok=True)

# File Size
def get_file_size_mb(filepath: Path) -> float:
    # Returns file size in MB.
    return filepath.stat().st_size / (1024 * 1024)

def save_figure(filepath):
    """
    Saves the current matplotlib figure.

    Parameters
    ----------
    filepath : Path
        Output path for the figure.
    """

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI)
    plt.close()