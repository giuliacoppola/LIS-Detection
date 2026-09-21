# train_model.py

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

# === Configurazione ===
DATASET_DIR = "handDataset2"
MODEL_KERAS = "lis_dnn_model_buono.keras"
LABEL_ENCODER_FILE = "final/label_encoder.pkl"
TEST_DATA_FILE = "final/test_data.npz"

# === Caricamento dataset ===
X, y = [], []
print("[INFO] Caricamento dati...")

for label_folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, label_folder)
    for root, _, files in os.walk(folder_path):
        vector_file = next((os.path.join(root, f) for f in files if f.startswith("vectors")), None)
        class_file = next((os.path.join(root, f) for f in files if f.startswith("classes")), None)
        if vector_file and class_file:
            with open(vector_file, "rb") as vf, open(class_file, "rb") as cf:
                vectors = pickle.load(vf)
                classes = pickle.load(cf)
            for v, c in zip(vectors, classes):
                if len(v) == 42:  # usa solo quelli con 42 feature
                    X.append(v)
                    y.append(c)

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"[INFO] Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")

# === Encoding etichette ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
with open(LABEL_ENCODER_FILE, "wb") as f:
    pickle.dump(encoder, f)

# === Normalizzazione ===
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# === Split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

np.savez(TEST_DATA_FILE, X_test=X_test, y_test=y_test, X_train=X_train)

print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# === Definizione del modello ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,), name="input"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("[INFO] Inizio addestramento...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

# === Salvataggio modello in formato Keras ===
model.save(MODEL_KERAS)
print(f"✅ Modello salvato come {MODEL_KERAS}")

# === Plot accuratezza ===
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuratezza")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# === quantizzazione ===

# === Configura il convertitore ===

# === Dataset di rappresentazione ===
def representative_dataset():
    for i in range(100):
        yield [X_train[i].astype(np.float32).reshape(1, -1)]

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
