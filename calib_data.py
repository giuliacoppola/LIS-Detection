import numpy as np

# === Carica X_test e prendi 100 campioni per calibrazione ===
data = np.load("final/test_data.npz")
X_test = data["X_test"]

# ⚠️ Attenzione: non usare questi 100 campioni per test finale
X_calib = X_test[:100]

# === Salva il file per la quantizzazione ===
np.save("X_calib.npy", X_calib)
print("✅ File X_calib.npy creato con 100 campioni.")
