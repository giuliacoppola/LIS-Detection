import cv2
import numpy as np
import tensorflow as tf

# === Modelli ===
palm_interpreter = tf.lite.Interpreter(model_path="models/palm_detection.tflite")
palm_interpreter.allocate_tensors()
palm_input = palm_interpreter.get_input_details()
palm_output = palm_interpreter.get_output_details()

landmark_interpreter = tf.lite.Interpreter(model_path="models/hand_landmark.tflite")
landmark_interpreter.allocate_tensors()
landmark_input = landmark_interpreter.get_input_details()
landmark_output = landmark_interpreter.get_output_details()

# Connessioni dita
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

cap = cv2.VideoCapture(0)
print("Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    palm_interpreter.set_tensor(palm_input[0]['index'], input_tensor)
    palm_interpreter.invoke()

    scores = palm_interpreter.get_tensor(palm_output[0]['index'])
    boxes = palm_interpreter.get_tensor(palm_output[1]['index'])

    max_idx = np.argmax(scores[0])
    max_score = scores[0][max_idx][0]
    print(f"[Palm] Max score: {max_score:.3f}")

    if max_score < 0.3:
        cv2.imshow("Webcam", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Bounding box
    cx, cy, w, h = boxes[0][max_idx][:4]
    img_h, img_w, _ = frame.shape
    xmin = int((cx - w/2) * img_w)
    ymin = int((cy - h/2) * img_h)
    xmax = int((cx + w/2) * img_w)
    ymax = int((cy + h/2) * img_h)

    padding = 20
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(img_w, xmax + padding)
    ymax = min(img_h, ymax + padding)

    if xmax <= xmin or ymax <= ymin:
        continue

    hand_crop = frame[ymin:ymax, xmin:xmax]
    if hand_crop.size == 0:
        continue

    # === Preprocess per landmark
    hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    hand_resized = cv2.resize(hand_rgb, (256, 256)).astype(np.float32) / 255.0
    input_landmark = np.expand_dims(hand_resized, axis=0)

    landmark_interpreter.set_tensor(landmark_input[0]['index'], input_landmark)
    landmark_interpreter.invoke()

    keypoints = landmark_interpreter.get_tensor(landmark_output[0]['index'])[0]
    flag = landmark_interpreter.get_tensor(landmark_output[1]['index'])[0][0]

    print(f"[DEBUG] Landmark flag: {flag:.2f}")

    if flag < 0.5:
        cv2.imshow("Webcam", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Converti keypoints da normalizzati a coordinate del frame
    points = []
    for i in range(21):
        x_norm = keypoints[i * 2]
        y_norm = keypoints[i * 2 + 1]
        x = int(x_norm * (xmax - xmin)) + xmin
        y = int(y_norm * (ymax - ymin)) + ymin
        points.append((x, y))
        cv2.circle(frame_copy, (x, y), 4, (0, 255, 255), -1)

    # Disegna connessioni
    for pt1, pt2 in connections:
        cv2.line(frame_copy, points[pt1], points[pt2], (255, 255, 0), 2)

    # Mostra risultato
    cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("Webcam", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
