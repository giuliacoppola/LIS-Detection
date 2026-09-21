import os
import pickle

# Percorso dataset
dataset_dir = "handDataset2"

total_vectors = 0
valid_vectors = 0
invalid_vectors = 0
invalid_details = []

print("[INFO] Controllo integrità dataset...\n")

for label_folder in sorted(os.listdir(dataset_dir)):
    folder_path = os.path.join(dataset_dir, label_folder)

    for root, dirs, files in os.walk(folder_path):
        vector_file = None
        class_file = None

        for f in files:
            if f.startswith("vectors"):
                vector_file = os.path.join(root, f)
            elif f.startswith("classes"):
                class_file = os.path.join(root, f)

        if vector_file and class_file:
            with open(vector_file, "rb") as vf:
                vectors = pickle.load(vf)
            with open(class_file, "rb") as cf:
                classes = pickle.load(cf)

            for idx, v in enumerate(vectors):
                total_vectors += 1
                if len(v) == 42:
                    valid_vectors += 1
                else:
                    invalid_vectors += 1
                    invalid_details.append({
                        "label": label_folder,
                        "file": vector_file,
                        "index": idx,
                        "length": len(v)
                    })

print("\n===== RISULTATI =====")
print(f"Totale vettori: {total_vectors}")
print(f"✅ Vettori validi (42 features): {valid_vectors}")
print(f"❌ Vettori invalidi: {invalid_vectors}")

if invalid_vectors > 0:
    print("\nDettagli vettori invalidi:")
    for d in invalid_details[:20]:  # Mostra massimo 20 per non esagerare
        print(f"  - Label: {d['label']}, File: {os.path.basename(d['file'])}, "
              f"Index: {d['index']}, Lunghezza: {d['length']}")

    print("\n⚠ Suggerimento: Rimuovi o correggi questi vettori prima di allenare il modello.")
else:
    print("✅ Tutti i vettori sono corretti (42 valori).")

print("=====================")
