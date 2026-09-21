import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# === Percorso al dataset ===
DATASET_DIR = "handDataset2"

# === Caricamento dati ===
X, y = [], []
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
                if len(v) == 42:
                    X.append(v)
                    y.append(c)

# === Encoding etichette ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# === Grafico distribuzione ===
unique, counts = np.unique(y_encoded, return_counts=True)
plt.figure(figsize=(10,5))
plt.bar(unique, counts, color='cornflowerblue')
plt.xticks(unique, [chr(65 + i) for i in unique])  # A–Z
plt.title("Distribuzione delle classi nel dataset LIS-Detection")
plt.xlabel("Lettera (classe)")
plt.ylabel("Numero di campioni")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
