# run_lis_repo_aligned_fixed.py
import cv2, numpy as np, tensorflow as tf

# ====== Modelli ======
PALM_TFLITE = "models/palm_detection.tflite"
LM_TFLITE   = "models/hand_landmark.tflite"
CLS_TFLITE  = "models/lis_dnn_model_quant_int8.tflite"

# ---- Interpreter ----
palm_interpreter = tf.lite.Interpreter(model_path=PALM_TFLITE)
palm_interpreter.allocate_tensors()
palm_in  = palm_interpreter.get_input_details()
palm_out = palm_interpreter.get_output_details()

lmk_interpreter  = tf.lite.Interpreter(model_path=LM_TFLITE)
lmk_interpreter.allocate_tensors()
lmk_in  = lmk_interpreter.get_input_details()
lmk_out = lmk_interpreter.get_output_details()

cls_interpreter  = tf.lite.Interpreter(model_path=CLS_TFLITE)
cls_interpreter.allocate_tensors()
cls_in  = cls_interpreter.get_input_details()[0]
cls_out = cls_interpreter.get_output_details()[0]
print("[CLS] input:", cls_in['dtype'], "quant:", cls_in['quantization'])
print("[CLS] output:", cls_out['dtype'], "quant:", cls_out['quantization'])

# ====== Anchors (BlazePalm 256) ======
def _scale_for_layer(min_s, max_s, lid, n):
    return (min_s + max_s) * 0.5 if n == 1 else (min_s + (max_s - min_s) * lid / (n - 1))

def make_anchors_blazepalm256():
    input_size = 256
    strides = (8, 16, 32)
    min_scale, max_scale = 0.1484375, 0.75
    anchors = []
    for li, s in enumerate(strides):
        fm = int(np.ceil(input_size / s))
        sc  = _scale_for_layer(min_scale, max_scale, li, len(strides))
        scn = _scale_for_layer(min_scale, max_scale, li + 1, len(strides)) if li != len(strides) - 1 else 1.0
        scales = [sc, scn] + ([np.sqrt(sc * scn)] if li == 1 else [])
        for y in range(fm):
            for x in range(fm):
                cx = (x + 0.5) / fm
                cy = (y + 0.5) / fm
                for sca in scales:
                    anchors.append([cx, cy, sca, sca])
    a = np.array(anchors, np.float32)
    assert a.shape[0] == 2944, f"Anchors attesi 2944, trovati {a.shape[0]}"
    return a

ANCHORS = make_anchors_blazepalm256()

# ====== Palm decode ======
def sigmoid_stable(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))

def iou_batch(a, b):
    inter = (np.minimum(a[2], b[:,2]) - np.maximum(a[0], b[:,0])).clip(0) * \
            (np.minimum(a[3], b[:,3]) - np.maximum(a[1], b[:,1])).clip(0)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter / (area_a + area_b - inter + 1e-6)

def nms(xyxy, scores, iou_th=0.3, topk=5):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size and len(keep) < topk:
        i = idx[0]; keep.append(i)
        if idx.size == 1: break
        idx = idx[1:][ iou_batch(xyxy[i], xyxy[idx[1:]]) < iou_th ]
    return keep

def decode_palm(classificators, regressors, anchors, score_th=0.3):
    scores_raw = classificators[0,:,0]
    scores = sigmoid_stable(scores_raw)  # palm output = logits

    reg = regressors[0,:,:]
    x_scale = y_scale = w_scale = h_scale = 256.0

    dets = []
    for i in range(anchors.shape[0]):
        s = scores[i]
        if s < score_th: continue
        axc, ayc, aw, ah = anchors[i]
        dx, dy, dw, dh = reg[i,:4]
        dw = np.clip(dw, -10.0, 10.0)
        dh = np.clip(dh, -10.0, 10.0)
        xc = dx / x_scale * aw + axc
        yc = dy / y_scale * ah + ayc
        w  = np.exp(dw / w_scale) * aw
        h  = np.exp(dh / h_scale) * ah
        dets.append([s, xc, yc, w, h])
    if not dets:
        return np.empty((0,5), np.float32)
    dets = np.array(dets, np.float32)
    xyxy = np.stack([dets[:,1]-dets[:,3]/2, dets[:,2]-dets[:,4]/2,
                     dets[:,1]+dets[:,3]/2, dets[:,2]+dets[:,4]/2], 1)
    keep = nms(xyxy, dets[:,0], 0.3, 5)
    return dets[keep]

# ====== Helpers ======
def dequant_logits(t, out_detail):
    if t.dtype == np.float32:
        return t
    s, z = out_detail['quantization']
    s = 1.0 if s == 0 else s
    return (t.astype(np.float32) - z) * s

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-6)

# ====== MAIN ======
cap = cv2.VideoCapture(0)
print("Premi 'q' per uscire.")
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    show = frame.copy()
    H, W = frame.shape[:2]

    # --- PALM ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, (256,256)).astype(np.float32) / 255.0
    palm_interpreter.set_tensor(palm_in[0]['index'], im[None, ...])
    palm_interpreter.invoke()
    cls_scores = palm_interpreter.get_tensor(palm_out[0]['index'])
    reg_data = palm_interpreter.get_tensor(palm_out[1]['index'])
    dets = decode_palm(cls_scores, reg_data, ANCHORS, score_th=0.3)

    if dets.shape[0] == 0:
        cv2.putText(show, "No hand", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow("Webcam", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
        continue

    # migliore detection
    s, xc, yc, w, h = dets[0]
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    pad = 20
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        frame_idx += 1
        continue

    hand = frame[y1:y2, x1:x2]
    if hand.size == 0:
        frame_idx += 1
        continue

    # --- LANDMARK ---
    hand_rgb = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    hand_resized = cv2.resize(hand_rgb, (256,256)).astype(np.float32) / 255.0
    lmk_interpreter.set_tensor(lmk_in[0]['index'], hand_resized[None, ...])
    lmk_interpreter.invoke()
    kps  = lmk_interpreter.get_tensor(lmk_out[0]['index'])[0]  # [42]
    flag = float(lmk_interpreter.get_tensor(lmk_out[1]['index'])[0][0])
    if flag < 0.5:
        cv2.putText(show, "Low landmark conf", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)
        cv2.imshow("Webcam", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
        continue

    # ---- Allineamento coordinate al repo ----
    feat42 = []
    crop_w = (x2 - x1)
    crop_h = (y2 - y1)
    for i in range(21):
        xn = float(kps[2*i])      # [0..1] nel crop
        yn = float(kps[2*i+1])
        x_abs = x1 + xn * crop_w  # pixel sul frame
        y_abs = y1 + yn * crop_h
        x_norm_frame = x_abs / W  # [0..1] sul frame
        y_norm_frame = y_abs / H
        feat42.extend([x_norm_frame, y_norm_frame])
        # opzionale: draw punti
        cv2.circle(show, (int(x_abs), int(y_abs)), 2, (0,255,255), -1)

    feat = np.array(feat42, dtype=np.float32).reshape(1, 42)

    # --- CLASSIFICATORE ---
    in_scale, in_zero = cls_in['quantization']
    in_scale = 1.0 if in_scale == 0 else in_scale
    qfeat = (feat / in_scale + in_zero).round().astype(cls_in['dtype'])
    cls_interpreter.set_tensor(cls_in['index'], qfeat)
    cls_interpreter.invoke()
    raw = cls_interpreter.get_tensor(cls_out['index'])[0]
    logits = dequant_logits(raw, cls_out)
    probs  = softmax(logits)

    pred = int(np.argmax(probs))
    letter = chr(ord('A') + pred)
    conf = float(probs[pred])

    # --- DRAW ---
    cv2.rectangle(show, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(show, f"{letter} ({conf:.2f})", (x1, max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Webcam", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
