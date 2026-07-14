import tensorflow as tf
from config import (BEST_MODEL_PATH,TFLITE_MODEL_PATH, CONVERSION_REPORT_PATH)
from utils import logger, save_json

def load_model():
    # Loads the trained Keras model.
    logger.info("Loading trained Keras model...")
    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    logger.info("Model loaded successfully.")
    return model

def convert_to_tflite(model):
    """
    Converts the Keras model to TensorFlow Lite.
    Returns
    -------
    bytes
        Serialized TensorFlow Lite model.
    """
    logger.info("Converting model to TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    tflite_model = converter.convert()
    logger.info("Model converted successfully.")

    return tflite_model

def save_tflite_model(tflite_model):
    # Saves the TensorFlow Lite model.
    logger.info("Saving TensorFlow Lite model...")

    with open(TFLITE_MODEL_PATH,"wb",) as file:
        file.write(tflite_model)

    logger.info(f"TFLite model saved to {TFLITE_MODEL_PATH}")

def verify_tflite_model():
    # Verifies that the generated TensorFlow Lite model can be loaded successfully.
    logger.info("Verifying TensorFlow Lite model...")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("TensorFlow Lite model verified successfully.")

    return {
        "input_shape": input_details[0]["shape"].tolist(),
        "output_shape": output_details[0]["shape"].tolist(),
    }

def generate_conversion_report(verification_info):
    # Generates a JSON report for the converted model.

    logger.info("Generating conversion report...")

    keras_size = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
    tflite_size = TFLITE_MODEL_PATH.stat().st_size / (1024 * 1024)

    report = {
        "keras_model_size_mb": round(keras_size, 2),
        "tflite_model_size_mb": round(tflite_size, 2),
        "size_reduction_percent": round(
            ((keras_size - tflite_size) / keras_size) * 100,
            2,
        ),
        "input_shape": verification_info["input_shape"],
        "output_shape": verification_info["output_shape"],
    }

    save_json(report,CONVERSION_REPORT_PATH)

    logger.info(f"Conversion report saved to {CONVERSION_REPORT_PATH}")

def main():
    # Complete TensorFlow Lite conversion pipeline.
    logger.info("=" * 60)
    logger.info("Starting TensorFlow Lite Conversion")
    logger.info("=" * 60)

    model = load_model()
    tflite_model = convert_to_tflite(model)

    save_tflite_model(tflite_model)

    verification_info = verify_tflite_model()

    generate_conversion_report(
        verification_info,
    )

    logger.info("=" * 60)
    logger.info("TensorFlow Lite conversion completed successfully.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()