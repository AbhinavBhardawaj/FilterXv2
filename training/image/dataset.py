"""
=========================================================
FilterX V2 - Dataset Utilities
=========================================================

Responsible for:
- Loading image datasets
- Dataset optimization
- Data augmentation
"""

import tensorflow as tf
from tensorflow.keras import layers

from utils import logger

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

# Data Augmentation
def get_data_augmentation():
    """
    Returns the data augmentation pipeline.
    Applied only during training.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

# Dataset Optimization
def configure_dataset(dataset, training=True):
    """
    Optimizes dataset pipeline.

    Parameters
    ----------
    dataset : tf.data.Dataset

    training : bool
        Whether dataset is used for training.

    Returns
    -------
    tf.data.Dataset
    """
    if training:
        dataset = dataset.shuffle(
            buffer_size=1000,
            seed=SEED,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Dataset Loader
def create_dataset(directory, shuffle):
    """
    Creates a TensorFlow dataset from a directory.

    Parameters
    ----------
    directory : Path
        Path to the dataset directory.

    shuffle : bool
        Whether to shuffle the dataset.

    Returns
    -------
    tf.data.Dataset
    """

    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="binary",
    )

# Load Train / Validation / Test
def load_datasets():
    """
    Loads train, validation and test datasets.

    Returns
    -------
    tuple
        (train_dataset, validation_dataset, test_dataset)
    """

    logger.info("Loading training dataset...")
    train_dataset = create_dataset(TRAIN_DIR,shuffle=True)

    logger.info("Loading validation dataset...")
    validation_dataset = create_dataset(VAL_DIR,shuffle=False)

    logger.info("Loading test dataset...")
    test_dataset = create_dataset(TEST_DIR,shuffle=False)

    train_dataset = configure_dataset(
        train_dataset,
        training=True,
    )

    validation_dataset = configure_dataset(
        validation_dataset,
        training=False,
    )

    test_dataset = configure_dataset(
        test_dataset,
        training=False,
    )

    logger.info("Datasets loaded successfully.")

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )

# Load Test Dataset Only
def load_test_dataset():
    """
    Loads only the test dataset.

    Used by evaluate.py to avoid loading
    train and validation datasets unnecessarily.

    Returns
    -------
    tf.data.Dataset
    """

    logger.info("Loading test dataset...")
    test_dataset = create_dataset(TEST_DIR,shuffle=False)

    test_dataset = configure_dataset(
        test_dataset,
        training=False,
    )

    logger.info("Test dataset loaded successfully.")
    return test_dataset