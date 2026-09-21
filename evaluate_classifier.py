import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Carica test set ===
data = np.load("final/test_data.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# === Carica modello ===
model = tf.keras.models.load_model("lis_dnn_model_buono.keras")

# === Carica label encoder ===
with open("final/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# === Predizione ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# === Report ===
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
