"""
=========================================================
FilterX V2 - Model Evaluation
=========================================================

Responsible for:
- Loading trained model
- Running predictions
- Computing evaluation metrics
- Generating evaluation reports
- Benchmarking inference
"""

from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
import pandas as pd

from dataset import load_test_dataset

from utils import (
    logger,
    save_json,
    save_model_summary,
    load_pickle,
    get_file_size_mb,
    save_figure,
)

from config import (
    BEST_MODEL_PATH,
    TRAINING_HISTORY_PATH,

    CLASSIFICATION_REPORT_PATH,
    METRICS_JSON_PATH,
    CONFUSION_MATRIX_PATH,
    CONFUSION_MATRIX_JSON_PATH,
    ROC_CURVE_PATH,
    PR_CURVE_PATH,
    TRAINING_HISTORY_DIR,
    MODEL_SUMMARY_PATH,
    THRESHOLD_ANALYSIS_PATH,
    INFERENCE_REPORT_PATH,

    DEFAULT_THRESHOLD,
    CLASS_NAMES,
    THRESHOLDS,
    FIGURE_SIZE,
    DPI,
    NUM_BENCHMARK_RUNS,
)

def load_model():
    # Load the best saved Keras model.
    logger.info("Loading trained model...")

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )
    logger.info("Model loaded successfully.")
    return model


def run_predictions(model,dataset):
    """
    Runs inference on the complete test dataset.

    Returns
    -------
    y_true : np.ndarray

    y_prob : np.ndarray

    y_pred : np.ndarray
    """

    logger.info("Running inference on test dataset...")

    y_true = []
    y_prob = []

    for images, labels in dataset:
        probabilities = model.predict(
            images,
            verbose=0,
        ).flatten()
        y_prob.extend(probabilities)
        y_true.extend(
            labels.numpy().flatten()
        )

    y_true = np.asarray(y_true)

    y_prob = np.asarray(y_prob)

    y_pred = (
        y_prob >= DEFAULT_THRESHOLD
    ).astype(np.int32)

    logger.info(f"Finished inference on {len(y_true)} images.")
    return y_true, y_prob, y_pred

def compute_metrics(y_true,y_prob,y_pred):
    """
    Computes all numerical evaluation metrics.

    Returns
    -------
    dict
    """
    logger.info("Computing evaluation metrics...")

    metrics = {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1_score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob,
            ),
    }

    logger.info("Metrics computed successfully.")
    return metrics

def save_metrics_json(metrics):
    # Saves evaluation metrics to a JSON file.
    logger.info("Saving metrics JSON...")

    save_json(
        metrics,
        METRICS_JSON_PATH,
    )

    logger.info(f"Metrics saved to {METRICS_JSON_PATH}")

def save_classification_report(y_true,y_pred):
    # Saves sklearn classification report.
    logger.info("Generating classification report...")

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    with open(CLASSIFICATION_REPORT_PATH,"w") as file:
        file.write(report)

    logger.info(f"Classification report saved to {CLASSIFICATION_REPORT_PATH}")

def save_confusion_matrix_json(y_true,y_pred):
    # Saves confusion matrix values as JSON.
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred,).ravel()

    matrix = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    save_json(matrix,CONFUSION_MATRIX_JSON_PATH)

def save_confusion_matrix(y_true,y_pred):
    # Saves confusion matrix plot.
    logger.info("Saving confusion matrix...")
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
    )

    image = ax.imshow(cm)
    plt.colorbar(image)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    save_figure(CONFUSION_MATRIX_PATH)
    logger.info("Confusion matrix saved.")

def plot_roc_curve(y_true,y_prob):
    # Saves ROC Curve.
    logger.info("Generating ROC curve...")

    fpr, tpr, _ = roc_curve(y_true,y_prob)
    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(fpr,tpr,
        label="ROC Curve"
    )

    plt.plot([0, 1],[0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    save_figure(ROC_CURVE_PATH)
    logger.info("ROC curve saved.")

def plot_precision_recall_curve(y_true,y_prob):
    # Saves Precision-Recall Curve.
    logger.info("Generating Precision-Recall curve...")

    precision, recall, _ = precision_recall_curve(y_true,y_prob,)
    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(recall,precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    save_figure(PR_CURVE_PATH)
    logger.info("Precision-Recall curve saved.")

def threshold_analysis(y_true,y_prob):
    # Evaluates model performance across multiple thresholds.
    logger.info("Running threshold analysis...")
    results = []

    for threshold in THRESHOLDS:
        predictions = (
            y_prob >= threshold
        ).astype(np.int32)

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    logger.info(f"Threshold analysis saved to {THRESHOLD_ANALYSIS_PATH}")

def benchmark_model(model):
    # Benchmarks inference latency.
    logger.info("Benchmarking model...")
    dummy_image = np.random.rand(
        1,
        224,
        224,
        3,
    ).astype(np.float32)

    timings = []

    for _ in range(NUM_BENCHMARK_RUNS):
        start = time.perf_counter()

        model.predict(
            dummy_image,
            verbose=0,
        )
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    report = {

        "average_latency_ms":
            float(np.mean(timings)),

        "min_latency_ms":
            float(np.min(timings)),

        "max_latency_ms":
            float(np.max(timings)),

        "std_latency_ms":
            float(np.std(timings)),

        "model_size_mb":
            get_file_size_mb(
                BEST_MODEL_PATH
            ),
    }
    save_json(report,INFERENCE_REPORT_PATH)
    logger.info(f"Inference report saved to {INFERENCE_REPORT_PATH}")

def export_model_summary(model):
    # Saves model architecture summary.
    logger.info("Saving model summary...")
    save_model_summary(model,MODEL_SUMMARY_PATH)

    logger.info("Model summary saved.")

def plot_metric(history,metric_name,save_path):
    # Plots a training metric along with its validation metric if available.
    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        history[metric_name],
        label=f"Train {metric_name}",
    )

    validation_metric = f"val_{metric_name}"

    if validation_metric in history:
        plt.plot(
            history[validation_metric],
            label=f"Validation {metric_name}",
        )

    plt.xlabel("Epoch")
    display_name = metric_name.replace("_", " ").title()
    plt.ylabel(display_name)
    plt.title(display_name)
    plt.legend()

    save_figure(save_path)

def plot_training_history():
    # Generates plots for all training metrics.
    logger.info("Generating training history plots...")
    history = load_pickle(TRAINING_HISTORY_PATH)

    metrics_to_plot = [
        metric
        for metric in history.keys()
        if not metric.startswith("val_")
    ]

    for metric in metrics_to_plot:
        save_path = (TRAINING_HISTORY_DIR /f"{metric}.png")
        plot_metric(
            history,
            metric,
            save_path,
        )

    logger.info("Training history plots generated successfully.")

def print_summary(metrics): 
    # Prints evaluation summary.

    logger.info("=" * 60)
    logger.info("FilterX Evaluation Summary")
    logger.info("=" * 60)

    for metric, value in metrics.items():
        logger.info(
            f"{metric:<15}: {value:.4f}"
        )

    logger.info("=" * 60)

def main():
    # Runs the complete evaluation pipeline.
    logger.info("Starting evaluation pipeline...")

    # Load model
    model = load_model()

    # Load dataset
    test_dataset = load_test_dataset()

    # Predictions
    y_true, y_prob, y_pred = run_predictions(
        model,
        test_dataset,
    )

    # Metrics
    metrics = compute_metrics(
        y_true,
        y_prob,
        y_pred,
    )
    save_metrics_json(metrics)
    print_summary(metrics)

    # Reports
    save_classification_report(y_true,y_pred)
    save_confusion_matrix(y_true,y_pred)
    save_confusion_matrix_json(y_true,y_pred)

    plot_roc_curve(
        y_true,
        y_prob,
    )

    plot_precision_recall_curve(
        y_true,
        y_prob,
    )

    # Threshold Analysis
    threshold_analysis(
        y_true,
        y_prob,
    )
    # Model Reports
    export_model_summary(
        model,
    )
    benchmark_model(
        model,
    )

    plot_training_history()
    logger.info("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
