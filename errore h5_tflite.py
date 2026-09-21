import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# === Carica test set ===
data = np.load("final/test_data.npz")
X_test = data["X_test"]

# === Carica modello H5 ===
model_h5 = tf.keras.models.load_model("lis_dnn_model.h5")
y_pred_h5 = model_h5.predict(X_test)

# === Carica modello TFLite INT8 ===
interpreter = tf.lite.Interpreter(model_path="final_project/lis_dnn_model_quant_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]["quantization"]


diffs = []
for i, x in enumerate(X_test[:100]):
    # H5
    softmax_h5 = y_pred_h5[i]

    # TFLite
    x_int8 = (x / input_scale + input_zero_point).astype(np.int8).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    softmax_tflite = interpreter.get_tensor(output_details[0]['index'])[0]

    # Differenza assoluta media
    diff = np.mean(np.abs(softmax_h5 - softmax_tflite))
    diffs.append(diff)

plt.plot(diffs)
plt.title("Errore medio assoluto tra output H5 e TFLite")
plt.xlabel("Campione")
plt.ylabel("Errore medio")
plt.show()
