# step3_with_scaler_bestofk.py
import os, pickle
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# === Percorsi modelli/dati ===
PALM_TFLITE       = "models/palm_detection.tflite"
LANDMARK_TFLITE   = "models/hand_landmark.tflite"
CLASSIFIER_TFLITE = "models/lis_dnn_model_quant_int8.tflite"   # cambia se serve
LABEL_ENCODER_PKL = "LIS-Detection/label_encoder.pkl"
SCALER_PKL        = "LIS-Detection/scaler.pkl"
DATASET_DIR       = "LIS-Detection/handDataset2"  # solo se ti serve altro in futuro

# === Parametri palm/landmark (come step2 best-of-k) ===
INPUT_RANGE        = "zero_one"
SCALE              = 1.35
OUT_SIZE           = 256

PALM_SCORE_TH      = 0.35
PALM_TOPK          = 5
HANDFLAG_GOOD_TH   = 0.50
HANDFLAG_MIN_ACC   = 0.25
REACQUIRE_EVERY    = 8
MAX_OUTSIDE_FRAC   = 0.20
MAX_JUMP_NORM      = 0.18
EMA_ALPHA          = 0.5
MEDIAN_WIN         = 5

USE_FACE_REJECT    = True
FACE_MIN_SIZE_FRAC = 0.18
FACE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# === Opzioni classificazione/visual ===
DISPLAY_MIRROR   = False      # specchia SOLO la finestra
MIRROR_FEATURES  = False      # specchia i dati (x' = W-1-x) PRIMA dello scaler  (toggle: tasto 'm')
SMOOTH_PROBS_N   = 8          # media mobile delle probabilità (0=off)
SHOW_TOP3        = True       # stampa top3 in console

# ---------- Anchors BlazePalm @256 (2944) ----------
def _scale_for_layer(min_s, max_s, lid, n):
    return (min_s + max_s) / 2 if n == 1 else min_s + (max_s - min_s) * lid / (n - 1)

def make_anchors():
    input_size = 256
    strides = (8, 16, 32)
    min_scale, max_scale = 0.1484375, 0.75
    anchors=[]
    for li, s in enumerate(strides):
        fm = int(np.ceil(input_size / s))
        sc  = _scale_for_layer(min_scale, max_scale, li, len(strides))
        scn = _scale_for_layer(min_scale, max_scale, li+1, len(strides)) if li != len(strides)-1 else 1.0
        scales = [sc, scn] + ([np.sqrt(sc*scn)] if li == 1 else [])
        for y in range(fm):
            for x in range(fm):
                cx=(x+0.5)/fm; cy=(y+0.5)/fm
                for sca in scales:
                    anchors.append([cx, cy, sca, sca])
    a = np.array(anchors, np.float32)
    assert a.shape[0] == 2944
    return a

ANCHORS = make_anchors()

# ---------- Utils ----------
def sigmoid_stable(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))

def decode_palm_with_keypoints_pos2(classificators, regressors, anchors, score_th=PALM_SCORE_TH):
    # atteso: [1,N,1] e [1,N,18]; se invertiti, scambiali
    if classificators.shape[-1] == 18 and regressors.shape[-1] == 1:
        classificators, regressors = regressors, classificators

    scores = sigmoid_stable(classificators[0, :, 0])
    reg    = regressors[0, :, :]

    dets, all_kp = [], []
    for i, (axc, ayc, aw, ah) in enumerate(anchors):
        s = scores[i]
        if s < score_th: continue
        dx, dy, dw, dh = reg[i, :4]
        xc = axc + (dx / 256.0)
        yc = ayc + (dy / 256.0)
        w  = np.exp(np.clip(dw, -10.0, 10.0) / 256.0) * aw
        h  = np.exp(np.clip(dh, -10.0, 10.0) / 256.0) * ah
        kps = []
        for j in range(7):
            kx_off = reg[i, 4 + 2*j]
            ky_off = reg[i, 4 + 2*j + 1]
            kx = axc + (kx_off / 256.0)
            ky = ayc + (ky_off / 256.0)
            kps.append([kx, ky])
        dets.append([s, xc, yc, w, h])
        all_kp.append(kps)

    if not dets:
        return np.empty((0,5), np.float32), np.empty((0,7,2), np.float32)
    dets   = np.array(dets, np.float32)
    all_kp = np.array(all_kp, np.float32)
    return dets, all_kp

def crop_centered(frame, det, kps, scale=SCALE, out_size=OUT_SIZE):
    H, W = frame.shape[:2]
    cx = (kps[:,0].mean()) * W
    cy = (kps[:,1].mean()) * H
    _, _, _, w, h = det
    side = int(max(w*W, h*H) * scale)
    x1 = int(cx - side/2); y1 = int(cy - side/2)
    x2 = x1 + side;        y2 = y1 + side
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, (0,0,0,0)
    crop256 = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop256, (x1, y1, x2, y2)

def run_landmark_dual(lm, lm_in, lm_out, crop_bgr_256):
    rgb = cv2.cvtColor(crop_bgr_256, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp01 = rgb / 255.0
    inp11 = inp01 * 2.0 - 1.0
    best = None
    for arr in (inp01, inp11):
        lm.set_tensor(lm_in[0]['index'], arr[None, ...])
        lm.invoke()
        k = lm.get_tensor(lm_out[0]['index'])[0].astype(np.float32)
        f = float(lm.get_tensor(lm_out[1]['index'])[0][0])
        if k.max() > 1.5:  # alcuni landmark danno *256
            k /= 256.0
        fprob = f if 0.0 <= f <= 1.0 else float(sigmoid_stable(f))
        if (best is None) or (fprob > best[0]):
            best = (fprob, k)
    return best  # (flag_prob, kpts_norm_crop[42])

def frac_outside01(k):
    k2 = k.reshape(-1,2)
    return np.logical_or(k2<0.0, k2>1.0).any(axis=1).mean()

def ema(prev, cur, a=EMA_ALPHA):
    return a*cur + (1.0-a)*prev

def softmax(z):
    z = z - np.max(z); e = np.exp(z); return e / (e.sum() + 1e-8)

def main():
    global MIRROR_FEATURES

    # --- TFLite interpreters ---
    palm = tf.lite.Interpreter(model_path=PALM_TFLITE);       palm.allocate_tensors()
    lm   = tf.lite.Interpreter(model_path=LANDMARK_TFLITE);   lm.allocate_tensors()
    clf  = tf.lite.Interpreter(model_path=CLASSIFIER_TFLITE); clf.allocate_tensors()
    palm_in, palm_out = palm.get_input_details(), palm.get_output_details()
    lm_in,   lm_out   = lm.get_input_details(),   lm.get_output_details()
    clf_in,  clf_out  = clf.get_input_details()[0], clf.get_output_details()[0]

    in_scale, in_zp   = clf_in['quantization']
    out_scale, out_zp = clf_out['quantization']
    print(f"[CLS] in_quant:  scale={in_scale} zp={in_zp}")
    print(f"[CLS] out_quant: scale={out_scale} zp={out_zp}")

    # --- labels & scaler ---
    classes = [chr(ord('A')+i) for i in range(26)]
    if os.path.exists(LABEL_ENCODER_PKL):
        with open(LABEL_ENCODER_PKL,"rb") as f:
            enc = pickle.load(f)
        classes = list(enc.classes_)
        print("[labels]", classes)
    if not os.path.exists(SCALER_PKL):
        raise FileNotFoundError("scaler.pkl non trovato: addestra e salva lo scaler nel training.")
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    print("[scaler] caricato")

    # --- face reject ---
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if USE_FACE_REJECT else None

    # --- stato tracking/smoothing landmarks ---
    last_roi = None
    last_k   = None
    k_hist   = deque(maxlen=MEDIAN_WIN)
    frame_id = 0
    lost_cnt = 0

    # --- smoothing classifier ---
    prob_hist = deque(maxlen=SMOOTH_PROBS_N) if SMOOTH_PROBS_N>0 else None

    cap = cv2.VideoCapture(0)
    print("Premi 'm' per mirror delle FEATURE, 'q' per uscire.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        vis = frame.copy()
        if DISPLAY_MIRROR: vis = cv2.flip(vis, 1)

        need_reacquire = (frame_id % REACQUIRE_EVERY == 0) or (last_roi is None) or (lost_cnt > 4)

        # --- face detection (anti-faccia) ---
        face_boxes=[]
        if USE_FACE_REJECT:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 4,
                minSize=(int(FACE_MIN_SIZE_FRAC*W), int(FACE_MIN_SIZE_FRAC*H))
            )
            for (fx,fy,fw,fh) in faces:
                face_boxes.append((fx,fy,fx+fw,fy+fh))
                cv2.rectangle(vis,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)

        best_candidate = None  # (flag, (x1,y1,x2,y2), kpts_norm_crop)

        if need_reacquire:
            # --- PALM infer ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im  = cv2.resize(rgb,(256,256)).astype(np.float32)/255.0
            if INPUT_RANGE == "neg1_1": im = im*2.0 - 1.0
            palm.set_tensor(palm_in[0]['index'], im[None,...]); palm.invoke()
            cls = palm.get_tensor(palm_out[0]['index'])
            reg = palm.get_tensor(palm_out[1]['index'])
            dets, kps = decode_palm_with_keypoints_pos2(cls, reg, ANCHORS)

            if dets.shape[0]:
                order = np.argsort(-dets[:,0])[:PALM_TOPK]
                for idx in order:
                    score, xc,yc,w,h = dets[idx]
                    if score < PALM_SCORE_TH: continue
                    crop256, (x1,y1,x2,y2) = crop_centered(frame, dets[idx], kps[idx], SCALE, OUT_SIZE)
                    if crop256 is None: continue

                    # scarta se overlap forte con faccia
                    if face_boxes:
                        discard=False
                        for (fx1,fy1,fx2,fy2) in face_boxes:
                            xx1 = max(x1,fx1); yy1 = max(y1,fy1)
                            xx2 = min(x2,fx2); yy2 = min(y2,fy2)
                            inter = max(0,xx2-xx1)*max(0,yy2-yy1)
                            if inter > 0.35*((x2-x1)*(y2-y1)):
                                discard=True; break
                        if discard: continue

                    flagp, k = run_landmark_dual(lm, lm_in, lm_out, crop256)
                    if flagp < HANDFLAG_MIN_ACC: continue
                    if frac_outside01(k) > MAX_OUTSIDE_FRAC: continue
                    if (best_candidate is None) or (flagp > best_candidate[0]):
                        best_candidate = (flagp, (x1,y1,x2,y2), k)

            if best_candidate:
                last_roi = best_candidate[1]
                last_k   = best_candidate[2]
                k_hist.clear(); k_hist.append(last_k.copy())
                lost_cnt = 0
            else:
                lost_cnt += 1

        # --- Landmark update + filtro ---
        pts = None
        if last_roi is not None:
            x1,y1,x2,y2 = last_roi
            x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
            if x2<=x1 or y2<=y1:
                last_roi=None; last_k=None; k_hist.clear()
            else:
                crop = frame[y1:y2, x1:x2]
                crop256 = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
                flagp, k = run_landmark_dual(lm, lm_in, lm_out, crop256)
                out_frac = frac_outside01(k)

                jump_ok = True
                if last_k is not None:
                    jump = np.abs(k - last_k).reshape(-1,2).max(axis=1)
                    if (jump > MAX_JUMP_NORM).mean() > 0.40:
                        jump_ok = False

                good = (flagp >= HANDFLAG_GOOD_TH) and (out_frac <= MAX_OUTSIDE_FRAC) and jump_ok
                if good:
                    k_hist.append(k.copy())
                    k_med = np.median(np.stack(k_hist,0), axis=0) if len(k_hist)>=3 else k
                    last_k = k_med if last_k is None else ema(last_k, k_med, EMA_ALPHA)
                    lost_cnt = 0
                else:
                    lost_cnt += 1
                    if lost_cnt >= 10:
                        last_roi=None; last_k=None; k_hist.clear()

                # disegno ROI
                if DISPLAY_MIRROR:
                    cv2.rectangle(vis,(W-1-x2,y1),(W-1-x1,y2),(0,255,0),2)
                else:
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)

                kk = last_k if last_k is not None else k
                pts=[]
                for i in range(21):
                    xn,yn = float(kk[2*i]), float(kk[2*i+1])
                    x = int(x1 + xn*(x2-x1)); y = int(y1 + yn*(y2-y1))
                    pts.append((x,y))
                    if DISPLAY_MIRROR:
                        cv2.circle(vis,(W-1-x,y),2,(0,255,255),-1)
                    else:
                        cv2.circle(vis,(x,y),2,(0,255,255),-1)

                cv2.putText(vis, f"flag:{flagp:.2f} out:{out_frac*100:.0f}% lost:{lost_cnt}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # --- CLASSIFICAZIONE solo se abbiamo pts ---
        if pts is not None:
            # feature in pixel (come nel dataset) + mirroring opzionale
            feats_px = np.empty(42, np.float32)
            for i,(x,y) in enumerate(pts):
                feats_px[2*i]   = (W - 1 - x) if MIRROR_FEATURES else x
                feats_px[2*i+1] = y

            feats_sc = scaler.transform([feats_px])[0].astype(np.float32)

            # TFLite infer
            if in_scale != 0.0:
                qfeat = np.clip(np.round(feats_sc / in_scale + in_zp), -128, 127).astype(np.int8).reshape(1,42)
                clf.set_tensor(clf_in['index'], qfeat)
            else:
                clf.set_tensor(clf_in['index'], feats_sc[None,:])
            clf.invoke()
            out = clf.get_tensor(clf_out['index'])[0]
            logits = (out.astype(np.int32)-out_zp)*out_scale if out_scale!=0.0 else out.astype(np.float32)
            probs  = softmax(logits)

            if prob_hist is not None:
                prob_hist.append(probs)
                probs = np.mean(np.stack(prob_hist,0), axis=0)

            topi = int(np.argmax(probs)); prob = float(probs[topi])
            label = classes[topi] if 0 <= topi < len(classes) else f"#{topi}"

            if SHOW_TOP3:
                top3 = probs.argsort()[-3:][::-1]
                print("TOP3:", [(classes[i], float(probs[i])) for i in top3],
                      "| mirror_features:", MIRROR_FEATURES)

            cv2.putText(vis, f"{label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3)
            cv2.putText(vis, f"{prob*100:.1f}%", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(vis, "No hand", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

        cv2.imshow("STEP3 - LIS (best-of-k + scaler)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            MIRROR_FEATURES = not MIRROR_FEATURES
            print(">> MIRROR_FEATURES =", MIRROR_FEATURES)
        elif k == ord('q'):
            break

        frame_id += 1

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
