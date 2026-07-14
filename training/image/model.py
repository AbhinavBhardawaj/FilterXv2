"""
=========================================================
FilterX V2 - Model Architecture
=========================================================

Responsible for:
- Building the MobileNetV2 model
- Preparing the model for fine tuning
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dataset import get_data_augmentation

from config import (
    IMAGE_SIZE,
    DROPOUT_RATE,
    DENSE_UNITS,
    UNFREEZE_LAST_N_LAYERS,
)

def build_model(): 
    # Builds the Stage 1 model with a frozen MobileNetV2 backbone.
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )

    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    augmentation = get_data_augmentation()
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        DENSE_UNITS,
        activation="swish"
    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1,activation="sigmoid")(x)

    model = Model(
        inputs,
        outputs,
        name="FilterX_MobileNetV2"
    )
    return model, base_model

def unfreeze_model(base_model):
    # Unfreezes the last N layers of MobileNetV2 for fine tuning.
    base_model.trainable = True

    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

# Compile Model
def compile_model(model, learning_rate):
    # Compiles the model with the given learning rate.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )