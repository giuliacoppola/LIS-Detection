import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score

# === File ===
KERAS_MODEL = "lis_dnn_model_buono.keras"
TFLITE_MODEL = "lis_dnn_model_quant_int8.tflite"
LABEL_ENCODER = "label_encoder.pkl"
TEST_DATA = "test_data.npz"

# === Carica test set e label encoder ===
data = np.load(TEST_DATA)
X_test = data["X_test"]
y_test = data["y_test"]

with open(LABEL_ENCODER, "rb") as f:
    encoder = pickle.load(f)

# === Carica modello Keras ===
model = tf.keras.models.load_model(KERAS_MODEL)
y_pred_keras = model.predict(X_test)
y_pred_keras_classes = np.argmax(y_pred_keras, axis=1)
acc_keras = accuracy_score(y_test, y_pred_keras_classes)

# === Carica modello TFLite quantizzato ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]["quantization"]

# === Inference TFLite ===
y_pred_tflite = []
for x in X_test:
    x_int8 = ((x / input_scale) + input_zero_point).astype(np.int8).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred_tflite.append(np.argmax(output))

y_pred_tflite = np.array(y_pred_tflite)
acc_tflite = accuracy_score(y_test, y_pred_tflite)

# === Risultati ===
print("\n📊 COMPARAZIONE ACCURACY MODELLI")
print(f"✅ Keras Model Accuracy     : {acc_keras * 100:.2f}%")
print(f"✅ TFLite INT8 Model Accuracy: {acc_tflite * 100:.2f}%")
