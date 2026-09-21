import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === File ===
MODEL_TFLITE = "lis_dnn_model.tflite"
LABEL_ENCODER_FILE = "final/label_encoder.pkl"
TEST_DATA_FILE = "final/test_data.npz"

# === Caricamento LabelEncoder e dati ===
print("[INFO] Caricamento etichette e dati...")
with open(LABEL_ENCODER_FILE, "rb") as f:
    encoder = pickle.load(f)

data = np.load(TEST_DATA_FILE)
X_test, y_test = data["X_test"], data["y_test"]

# === Caricamento modello TFLite ===
print("[INFO] Caricamento modello TFLite...")
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()

# Ottieni dettagli input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"[DEBUG] Input shape: {input_details[0]['shape']}")
print(f"[DEBUG] Output shape: {output_details[0]['shape']}")

# === Predizione sul test set ===
print("[INFO] Predizioni sul modello quantizzato...")
y_pred = []

for x in X_test:
    x = np.expand_dims(x, axis=0).astype(np.float32)  # shape (1, 42)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(np.argmax(output_data))

y_pred = np.array(y_pred)

# === Accuracy e Report ===
accuracy = (y_pred == y_test).mean() * 100
print("\n===== RISULTATI TEST (TFLITE) =====")
print(f"Accuracy: {accuracy:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# === Matrice di Confusione ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Matrice di Confusione - Modello Quantizzato")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()
