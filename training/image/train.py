"""
=========================================================
FilterX V2 - Image Training Pipeline
=========================================================

Responsibilities:
- Load datasets
- Build model
- Compile model
- Train Stage 1
- Fine Tune Stage 2
- Save history
- Save final model
"""
import time
from dataset import load_datasets
from model import (
    build_model,
    unfreeze_model,
    compile_model,
)
from callbacks import get_callbacks
from config import (
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    LEARNING_RATE,
    FINE_TUNE_LR,
    FINAL_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    SEED,
    BEST_MODEL_PATH,
)
from utils import (logger, save_pickle, set_seed,save_model_summary)

# Merge Histories
def merge_histories(history1, history2):
    # Merge Stage 1 and Stage 2 training histories.
    merged = {}

    all_keys = set(history1.history) | set(history2.history)

    for key in all_keys:
        merged[key] = (history1.history.get(key, []) + history2.history.get(key, []))

    return merged

# Main Training Function
def main():
    start_time = time.time()
    set_seed(SEED)
    logger.info("=" * 70)
    logger.info("Loading datasets...")
    logger.info("=" * 70)

    train_ds, val_ds, _ = load_datasets()

    logger.info("=" * 70)
    logger.info("Building MobileNetV2...")
    logger.info("=" * 70)

    model, base_model = build_model()
    save_model_summary(model)

    # Stage 1
    logger.info("Starting Stage 1 Training...")

    compile_model(
        model,
        LEARNING_RATE,
    )

    callbacks = get_callbacks()

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Stage 2
    logger.info("Unfreezing model for fine tuning...")
    unfreeze_model(base_model)

    compile_model(
        model,
        FINE_TUNE_LR,
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=INITIAL_EPOCHS,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save Final Model
    logger.info("Saving final model...")
    model.save(FINAL_MODEL_PATH)

    # Save Training History
    merged_history = merge_histories(
        history_stage1,
        history_stage2,
    )

    save_pickle(merged_history,TRAINING_HISTORY_PATH)

    logger.info("=" * 70)
    logger.info("Training Complete")
    logger.info("=" * 70)

    logger.info(f"Best Model  : {BEST_MODEL_PATH}")
    logger.info(f"Final Model : {FINAL_MODEL_PATH}")
    logger.info(f"History     : {TRAINING_HISTORY_PATH}")

    elapsed = time.time() - start_time

    logger.info(f"Training Time: {elapsed/60:.2f} minutes")
if __name__ == "__main__":
    main()