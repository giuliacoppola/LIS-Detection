import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# === Percorsi ai file ===
MODEL_FILE = "final_project/lis_dnn_model_buono.keras"
DATA_FILE = "final/test_data.npz"
LABEL_ENCODER_FILE = "final/label_encoder.pkl"

# === Caricamento modello addestrato ===
print("[INFO] Caricamento modello...")
model = tf.keras.models.load_model(MODEL_FILE)

# === Caricamento dati di test ===
print("[INFO] Caricamento dati di test...")
data = np.load(DATA_FILE)
X_test = data["X_test"]
y_test = data["y_test"]

# === Caricamento encoder per ottenere le etichette originali ===
with open(LABEL_ENCODER_FILE, "rb") as f:
    encoder = pickle.load(f)

# === Predizione sul test set ===
print("[INFO] Predizione sul test set...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# === Creazione matrice di confusione ===
print("[INFO] Creazione matrice di confusione...")
cm = confusion_matrix(y_test, y_pred_classes)

# === Visualizzazione ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical", colorbar=False)
plt.title("Matrice di Confusione")
plt.xlabel("Classe Predetta")
plt.ylabel("Classe Reale")
plt.tight_layout()
plt.show()

# === Salvataggio opzionale dell’immagine ===
plt.savefig("matrice_confusione.png", dpi=300)
print("✅ Matrice salvata come 'matrice_confusione.png'")

