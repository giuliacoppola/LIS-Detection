import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === File ===
MODEL_H5 = "lis_dnn_model.h5"
LABEL_ENCODER_FILE = "final/label_encoder.pkl"
TEST_DATA_FILE = "final/test_data.npz"

# === Carica modello e dati ===
print("[INFO] Caricamento modello e dati...")
model = tf.keras.models.load_model(MODEL_H5)

with open(LABEL_ENCODER_FILE, "rb") as f:
    encoder = pickle.load(f)

data = np.load(TEST_DATA_FILE)
X_test, y_test = data["X_test"], data["y_test"]

# === Predizione ===
print("[INFO] Calcolo predizioni...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# === Accuracy e Report ===
print("\n===== RISULTATI TEST =====")
print(f"Accuracy: {(y_pred == y_test).mean() * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Matrice di Confusione")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()
