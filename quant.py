# quantize_model.py

import tensorflow as tf
import numpy as np

# === Caricamento modello  ===
model = tf.keras.models.load_model("lis_dnn_model.h5")
data = np.load("final/test_data.npz")
X_train = data["X_train"]

# === Dataset di rappresentazione ===
def representative_dataset():
    for i in range(100):
        yield [X_train[i].astype(np.float32).reshape(1, -1)]

# === Configura il convertitore ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# === Conversione ===
tflite_model = converter.convert()

# === Salvataggio modello quantizzato ===
with open("final_project/lis_dnn_model_quant_2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modello quantizzato INT8 salvato come lis_dnn_model_quant_int8.tflite")
