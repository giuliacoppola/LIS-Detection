import pickle
import os

# Percorso alla cartella della lettera E
folder_path = os.path.join("handDataset2", "E", "right")

# File da correggere (cercali dentro folder_path)
vector_file = None
class_file = None

for f in os.listdir(folder_path):
    if f.startswith("vectors"):
        vector_file = os.path.join(folder_path, f)
    elif f.startswith("classes"):
        class_file = os.path.join(folder_path, f)

if not vector_file or not class_file:
    raise FileNotFoundError("File vectors o classes non trovati nella cartella E/right")

print(f"[INFO] File trovati:\n- {vector_file}\n- {class_file}")

# Carica dati
with open(vector_file, "rb") as vf:
    vectors = pickle.load(vf)
with open(class_file, "rb") as cf:
    classes = pickle.load(cf)

print(f"[INFO] Numero vettori prima della correzione: {len(vectors)}")

# Rimuovi vettore corrotto (index 428)
invalid_index = 428
if len(vectors[invalid_index]) != 42:
    print(f"[INFO] Rimuovo vettore index {invalid_index}, lunghezza {len(vectors[invalid_index])}")
    vectors.pop(invalid_index)
    classes.pop(invalid_index)
else:
    print("[INFO] Nessun vettore corrotto trovato all'indice specificato.")

# Salva i file corretti
with open(vector_file, "wb") as vf:
    pickle.dump(vectors, vf)
with open(class_file, "wb") as cf:
    pickle.dump(classes, cf)

print(f"[INFO] Correzione completata. Numero vettori dopo: {len(vectors)}")
