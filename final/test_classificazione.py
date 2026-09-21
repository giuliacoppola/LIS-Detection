import numpy as np
import serial
import time

# --- Connessione seriale con la scheda STM32 ---
ser = serial.Serial(
    port="COM3",
    baudrate=115200,
    timeout=3  # tempo massimo di attesa per la risposta
)

# --- Prepara i dati da inviare ---
x_q = np.array([[5,212,0,216,18,207,39,183,26,149,
                 59,254,69,218,57,202,54,169,86,255,
                 122,165,121,108,129,78,106,252,168,
                 155,181,109,168,83,115,250,182,181,
                 212,139,231,114]])

# Converte in stringa con ":" e termina con newline (\n)
message = ":".join(str(x) for x in x_q[0]) + "\n"
print("Sending:", message)

# --- Invia i dati ---
ser.write(message.encode("utf-8"))

# --- Attendi la risposta ---
time.sleep(1.5)  # piccolo delay per dare tempo alla board di elaborare
response = ser.read_all().decode(errors='ignore').strip()

# --- Mostra la risposta ---
if response:
    print("Received from STM32:")
    prob = [int(x) for x in response.split(':') if x]
    print(prob)
else:
    print("⚠️ Nessuna risposta ricevuta dalla STM32")

# --- Chiudi la porta ---
ser.close()
