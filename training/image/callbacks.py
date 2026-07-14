"""
=========================================================
FilterX V2 - Training Callbacks
=========================================================

Responsible for:
- Saving the best model
- Early stopping
- Learning rate scheduling
- Training logs
"""

import tensorflow as tf

from config import (
    BEST_MODEL_PATH,
    MONITOR_METRIC,
    MONITOR_MODE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    TRAINING_LOG_PATH,
)

def get_callbacks():
    # Returns all callbacks used during training.
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            TRAINING_LOG_PATH,
            append=False,
        ),
    ]
    return callbacks