import cv2
import numpy as np
import tensorflow as tf

# === Carica modelli ===
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

palm_model = load_model("models/palm_detection.tflite")
landmark_model = load_model("models/hand_landmark.tflite")
lis_model = load_model("LIS-Detection/final_project/lis_dnn_model_quant_int8.tflite")

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return [interpreter.get_tensor(out['index']) for out in output_details]

# === Videocamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Errore apertura webcam")
    exit()

print("Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    input_palm = np.expand_dims(img_resized, axis=0)

    # === Inference palm detection ===
    palm_output = run_inference(palm_model, input_palm)
    scores = palm_output[0][0]
    boxes = palm_output[1][0]

    max_score_idx = np.argmax(scores)
    score = float(scores[max_score_idx])
    print(f"[Palm] Max score: {score:.2f}")

    if score < 0.4:
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === Bounding box ===
    box = boxes[max_score_idx]
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    img_h, img_w, _ = frame.shape
    xmin = int((cx - w/2) * img_w)
    ymin = int((cy - h/2) * img_h)
    xmax = int((cx + w/2) * img_w)
    ymax = int((cy + h/2) * img_h)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)

    if xmax <= xmin or ymax <= ymin:
        print("[Palm] Bounding box non valido.")
        continue

    hand_crop = frame[ymin:ymax, xmin:xmax]
    if hand_crop.size == 0:
        print("[Palm] Crop vuoto.")
        continue

    # === Inference landmark ===
    hand_resized = cv2.resize(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
    input_landmark = np.expand_dims(hand_resized, axis=0)
    landmark_output = run_inference(landmark_model, input_landmark)

    keypoints = landmark_output[0][0]
    flag = float(landmark_output[1][0][0])
    print(f"[Landmark] Hand flag: {flag:.2f}")

    if flag < 0.4:
        print("[Landmark] Mano non rilevata.")
        continue

    landmarks = keypoints.reshape((21, 2))
    for x, y in landmarks:
        px = int(x * (xmax - xmin)) + xmin
        py = int(y * (ymax - ymin)) + ymin
        cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)

    # === Inference LIS ===
    keypoints_scaled = np.clip(keypoints, 0, 1)
    keypoints_int8 = ((keypoints_scaled * 255) - 128).astype(np.int8)
    input_lis = np.expand_dims(keypoints_int8, axis=0)
    lis_out = run_inference(lis_model, input_lis)[0]
    prediction = int(np.argmax(lis_out))
    label = chr(ord('A') + prediction)
    print(f"[LIS] Predizione: {label}")

    # === Visualizza ===
    cv2.putText(frame, f"Pred: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
