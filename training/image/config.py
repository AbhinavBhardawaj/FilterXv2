"""
=========================================================
FilterX V2 - Image Training Configuration
=========================================================

This file contains all configurable parameters used during
training, evaluation, and model conversion.

Changing hyperparameters or paths should only require
editing this file.
"""

from pathlib import Path
# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset
DATASET_DIR = PROJECT_ROOT / "datasets" / "image"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Models
MODEL_DIR = PROJECT_ROOT / "models" / "image"

# Reports
REPORT_DIR = PROJECT_ROOT / "reports" / "image"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

THRESHOLDS = [i / 10 for i in range(1, 10)]

# Evaluation
FIGURE_SIZE = (8,6)
DPI = 150

# Evaluation Reports
METRICS_JSON_PATH = REPORT_DIR / "metrics.json"
THRESHOLD_ANALYSIS_PATH = REPORT_DIR / "threshold_analysis.csv"
MODEL_SUMMARY_PATH = REPORT_DIR / "model_summary.txt"
INFERENCE_REPORT_PATH = REPORT_DIR / "inference_report.json"
CONFUSION_MATRIX_JSON_PATH = REPORT_DIR / "confusion_matrix.json"

# Benchmark
NUM_BENCHMARK_RUNS = 100

# Random seed for reproducibility
SEED = 42

# Class labels
CLASS_NAMES = ["safe", "nsfw"]

# Stage 1 Training
INITIAL_EPOCHS = 40
LEARNING_RATE = 1e-3

# Stage 2 Fine Tuning
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 5e-6

# Number of layers to unfreeze from the end
UNFREEZE_LAST_N_LAYERS = 15

# Model Architecture
DROPOUT_RATE = 0.4
DENSE_UNITS = 128

# Callbacks
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.2

# We optimize for AUC because it is a better metric for
# binary classifiers like NSFW detection.
MONITOR_METRIC = "val_auc"
MONITOR_MODE = "max"

# Prediction
DEFAULT_THRESHOLD = 0.2

# Saved Models
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODEL_DIR / "filterx.keras"
TFLITE_MODEL_PATH = MODEL_DIR / "filterx.tflite"

# Training Artifacts
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.pkl"

# Reports
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORT_DIR / "roc_curve.png"
PR_CURVE_PATH = REPORT_DIR / "precision_recall_curve.png"

TRAINING_HISTORY_DIR = REPORT_DIR / "training_history"
TRAINING_HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAINING_LOG_PATH = REPORT_DIR / "training_log.csv"

# Conversion Report
CONVERSION_REPORT_PATH = REPORT_DIR / "tflite_conversion_report.json"