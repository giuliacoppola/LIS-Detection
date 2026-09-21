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
SCALER_FILE = "final/scaler.pkl"
TEST_DATA_FILE = "final/test_data.npz"
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Caricamento dataset ===
X, y = [], []
print("[INFO] Caricamento dati...")

for label_folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, label_folder)
    if not os.path.isdir(folder_path):
        continue
    for root, _, files in os.walk(folder_path):
        vector_file = next((os.path.join(root, f) for f in files if f.startswith("vectors")), None)
        class_file  = next((os.path.join(root, f) for f in files if f.startswith("classes")), None)
        if vector_file and class_file:
            with open(vector_file, "rb") as vf, open(class_file, "rb") as cf:
                vectors = pickle.load(vf)
                classes = pickle.load(cf)
            for v, c in zip(vectors, classes):
                # usa solo quelli con 42 feature
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 42:
                    X.append(np.asarray(v, dtype=np.float32))
                    y.append(c)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
assert X.ndim == 2 and X.shape[1] == 42, "Le feature devono avere shape [N,42]"
print(f"[INFO] Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")

# === Encoding etichette ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
with open(LABEL_ENCODER_FILE, "wb") as f:
    pickle.dump(encoder, f)
print(f"[OK] Salvato LabelEncoder in {LABEL_ENCODER_FILE}")
print("[INFO] Classi:", list(encoder.classes_))

# === Normalizzazione (IMPORTANTISSIMO: salviamo lo SCALER) ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)
print(f"[OK] Salvato MinMaxScaler in {SCALER_FILE}")

# === Split ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

np.savez(TEST_DATA_FILE, X_test=X_test, y_test=y_test, X_train=X_train)
print(f"[OK] Salvato split in {TEST_DATA_FILE}")
print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# === Modello ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,), name="input"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# callback: ferma se non migliora e salva il best
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_KERAS, monitor="val_accuracy", save_best_only=True
    )
]

print("[INFO] Inizio addestramento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# === Salvataggio modello (best già salvato da ModelCheckpoint) ===
# Salvo comunque lo stato corrente come sicurezza
model.save(MODEL_KERAS, overwrite=True)
print(f"✅ Modello salvato come {MODEL_KERAS}")

# === Valutazione su test ===
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[TEST] accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

# === Plot accuratezza ===
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuratezza")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
