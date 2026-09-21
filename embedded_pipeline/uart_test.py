import serial

# Configura la porta seriale
ser = serial.Serial(
    port='COM3',
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

print(f"Connesso a {ser.port} a {ser.baudrate} baud.")
print("Scrivi un carattere e premi INVIO (q per uscire)\n")

while True:
    # Leggi un carattere dalla tastiera
    ch = input(">>> ")

    if not ch:
        continue
    if ch.lower() == 'q':
        break

    # Invia solo il primo carattere digitato
    ser.write(ch[0].encode('utf-8'))

    # Legge la risposta dalla scheda
    reply = ser.readline().decode('utf-8').strip()
    if reply:
        print(f"<<< {reply}")
    else:
        print("<<< Nessuna risposta")

ser.close()
print("\nConnessione chiusa.")
