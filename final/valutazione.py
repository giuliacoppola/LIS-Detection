#!/usr/bin/env python
# coding: utf-8

"""
Script di debug per il classificatore Keras del progetto LIS.
Serve a verificare che il modello, lo scaler e l’encoder funzionino
correttamente, e che le predizioni siano consistenti.
"""

import os
import pickle
import numpy as np
from tensorflow import keras


# === CONFIGURAZIONE ===
MODEL_PATH = "lis_dnn_model.keras"  # percorso del modello Keras
SCALER_PATH = "scaler.pkl"              # scaler usato nel training
ENCODER_PATH = "label_encoder.pkl"      # label encoder usato nel training

# Se sai quanti input (es. 42 = 21 keypoint * 2 coordinate)
INPUT_SIZE = 42


def load_scaler(path):
    try:
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        print(f"[✔] Scaler caricato da {path}")
        return scaler
    except Exception as e:
        print(f"[!] Nessuno scaler caricato ({e})")
        return None


def load_encoder(path):
    try:
        with open(path, "rb") as f:
            encoder = pickle.load(f)
        print(f"[✔] Encoder caricato da {path}")
        print(f"    Classi: {list(encoder.classes_)}")
        return encoder
    except Exception as e:
        print(f"[!] Nessun encoder caricato ({e})")
        return None


def main():
    print("\n=== DEBUG CLASSIFICATORE KERAS ===")

    # 1️⃣ Carica modello
    if not os.path.exists(MODEL_PATH):
        print(f"[❌] Modello non trovato: {MODEL_PATH}")
        return
    model = keras.models.load_model(MODEL_PATH)
    print(f"[✔] Modello caricato: {MODEL_PATH}")

    # 2️⃣ Carica scaler ed encoder
    scaler = load_scaler(SCALER_PATH)
    encoder = load_encoder(ENCODER_PATH)

    # 3️⃣ Mostra informazioni sul modello
    print("\n=== INFO MODELLO ===")
    model.summary()

    # 4️⃣ Crea un vettore casuale per test
    x = np.random.rand(1, INPUT_SIZE).astype(np.float32)
    print("\nShape input:", x.shape)

    if scaler is not None:
        x_scaled = scaler.transform(x)
        print("Esempio di input scalato:", x_scaled[0, :5])
    else:
        x_scaled = x
        print("⚠️ Nessuno scaler: uso valori grezzi")

    # 5️⃣ Predizione
    probs = model.predict(x_scaled, verbose=0)
    print("\nOutput raw:", probs[0][:5], "...")
    print("Somma probabilità:", np.sum(probs[0]))

    # 6️⃣ Se non è normalizzato, applica softmax
    if np.sum(probs[0]) > 1.1:
        from tensorflow.nn import softmax
        probs = softmax(probs).numpy()
        print("\n(Softmax applicata manualmente)")
        print("Somma probabilità dopo softmax:", np.sum(probs[0]))

    # 7️⃣ Stampa predizione finale
    pred_idx = int(np.argmax(probs[0]))
    conf = float(probs[0][pred_idx])
    if encoder is not None:
        label = encoder.inverse_transform([pred_idx])[0]
    else:
        label = str(pred_idx)

    print(f"\n🧠 Predizione: {label} (confidenza {conf:.2f})")

    print("\n=== DEBUG COMPLETATO ===")


if __name__ == "__main__":
    main()
