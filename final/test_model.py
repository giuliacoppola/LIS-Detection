import numpy as np
from tensorflow import keras

print("📂 Caricamento modello...")
model = keras.models.load_model("lis_dnn_model_buono.keras")
print("✅ Modello caricato!")

# Stampa un riassunto del modello
print("\n📋 Sommario del modello:")
model.summary()

# Proviamo a vedere che input si aspetta
print("\n📐 Input shape del modello:", model.input_shape)

# Crea un vettore finto con 42 feature
fake_input = np.random.rand(1, 42)

print("\n🔮 Avvio predizione...")
pred_probs = model.predict(fake_input, verbose=1)
pred_class = np.argmax(pred_probs, axis=1)[0]

print("\n🔢 Predizione (classe):", pred_class)
print("📊 Probabilità:", pred_probs)
