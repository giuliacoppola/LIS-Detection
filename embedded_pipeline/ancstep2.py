# step2_palm_landmark_best_of_k.py
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# === importa dal tuo step1 che funziona ===
from ancstep1 import (
    PALM_TFLITE, INPUT_RANGE, SCALE, OUT_SIZE,
    ANCHORS, decode_palm_with_keypoints_pos2, crop_centered
)

LANDMARK_TFLITE = "models/hand_landmark.tflite"

# ---------- parametri ----------
PALM_SCORE_TH      = 0.35     # soglia minima proposta palm
PALM_TOPK          = 5        # quante proposte verificare col landmark
NMS_IOU_TH         = 0.30     # già usata in ancstep1->decode se serve
HANDFLAG_GOOD_TH   = 0.50     # scegliamo la proposta col flag migliore
HANDFLAG_MIN_ACC   = 0.25     # scarta del tutto se sotto questa
REACQUIRE_EVERY    = 8        # ogni N frame forziamo ri-aggancio col palm
TRACK_MAX_DIST     = 0.25     # priorità alle box vicine alla ROI precedente (in frazione lato img)

EMA_ALPHA          = 0.5      # smoothing esponenziale dei kpts
MEDIAN_WIN         = 5        # mediana locale
MAX_OUTSIDE_FRAC   = 0.20     # % max di keypoints fuori [0,1] (crop) prima di scartare
MAX_JUMP_NORM      = 0.18     # salto massimo ammesso tra frame consecutivi nel crop

USE_FACE_REJECT    = True     # filtro anti-faccia
FACE_MIN_SIZE_FRAC = 0.18     # faccia tipicamente >~18% del lato immagine
FACE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------- utils ----------
def sigmoid_stable(x):
    x = np.clip(x, -20, 20)
    return 1.0/(1.0+np.exp(-x))

def run_landmark_dual(lm, lm_in, lm_out, crop_bgr_256):
    """Prova [0,1] e [-1,1], restituisce (flag_prob, kpts_norm_crop[42]) migliore."""
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
    return best

def frac_outside01(k):
    k2 = k.reshape(-1,2)
    return np.logical_or(k2<0.0, k2>1.0).any(axis=1).mean()

def ema(prev, cur, a=EMA_ALPHA):
    return a*cur + (1.0-a)*prev

def bbox_center(x1,y1,x2,y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def l2_norm(a,b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def main():
    # interpreters
    palm = tf.lite.Interpreter(model_path=PALM_TFLITE);     palm.allocate_tensors()
    lm   = tf.lite.Interpreter(model_path=LANDMARK_TFLITE); lm.allocate_tensors()
    palm_in, palm_out = palm.get_input_details(), palm.get_output_details()
    lm_in,   lm_out   = lm.get_input_details(),   lm.get_output_details()

    # anti-faccia
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if USE_FACE_REJECT else None

    cap = cv2.VideoCapture(0)
    print("Premi 'q' per uscire.")

    last_roi = None                  # (x1,y1,x2,y2)
    last_k   = None                  # 42 norm in crop, filtrati
    k_hist   = deque(maxlen=MEDIAN_WIN)
    frame_id = 0
    lost_cnt = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        vis = frame.copy()

        need_reacquire = (frame_id % REACQUIRE_EVERY == 0) or (last_roi is None) or (lost_cnt > 4)

        # --- Face reject region (opzionale) ---
        face_boxes = []
        if USE_FACE_REJECT:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(int(FACE_MIN_SIZE_FRAC*W), int(FACE_MIN_SIZE_FRAC*H)))
            for (fx,fy,fw,fh) in faces:
                face_boxes.append((fx,fy,fx+fw,fy+fh))
                cv2.rectangle(vis,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)

        best_candidate = None  # (flag, (x1,y1,x2,y2), kpts_norm_crop[42])

        if need_reacquire:
            # --- PALM ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im  = cv2.resize(rgb,(256,256)).astype(np.float32)/255.0
            if INPUT_RANGE == "neg1_1": im = im*2.0 - 1.0
            palm.set_tensor(palm_in[0]['index'], im[None,...]); palm.invoke()
            cls = palm.get_tensor(palm_out[0]['index'])
            reg = palm.get_tensor(palm_out[1]['index'])
            dets, kps = decode_palm_with_keypoints_pos2(cls, reg, ANCHORS)

            if dets.shape[0]:
                # ordina per score, prendi top-K
                order = np.argsort(-dets[:,0])[:PALM_TOPK]
                # priorità alla box vicina alla ROI precedente
                if last_roi is not None:
                    (cxp, cyp) = bbox_center(*last_roi)
                    # converti in px per distanza
                    centers = []
                    for i in order:
                        s, xc,yc,w,h = dets[i]
                        centers.append((i, (int(xc*W), int(yc*H)), s))
                    dists = [ (i, l2_norm((cxp,cyp), c)) for (i,c,_) in centers ]
                    order = np.array([i for i,_ in sorted(dists, key=lambda z:z[1])])

                # valuta ogni proposta col landmark e scegli la migliore per flag
                for idx in order:
                    score, xc,yc,w,h = dets[idx]
                    if score < PALM_SCORE_TH:
                        continue
                    # box in px
                    bw, bh = int(w*W), int(h*H)
                    x1 = int(xc*W - bw/2); y1 = int(yc*H - bh/2)
                    x2 = x1 + bw;          y2 = y1 + bh
                    # crop centrato (margine SCALE)
                    crop256, (x1,y1,x2,y2) = crop_centered(frame, dets[idx], kps[idx], SCALE, OUT_SIZE)
                    if crop256 is None: continue

                    # scarta se overlap forte con faccia
                    discard_face = False
                    if face_boxes:
                        for (fx1,fy1,fx2,fy2) in face_boxes:
                            xx1 = max(x1,fx1); yy1 = max(y1,fy1)
                            xx2 = min(x2,fx2); yy2 = min(y2,fy2)
                            inter = max(0,xx2-xx1)*max(0,yy2-yy1)
                            if inter > 0.35*((x2-x1)*(y2-y1)):  # 35% overlap con faccia → troppo sospetto
                                discard_face = True; break
                    if discard_face: continue

                    flagp, k = run_landmark_dual(lm, lm_in, lm_out, crop256)
                    if flagp < HANDFLAG_MIN_ACC:
                        continue
                    if frac_outside01(k) > MAX_OUTSIDE_FRAC:
                        continue

                    if best_candidate is None or flagp > best_candidate[0]:
                        best_candidate = (flagp, (x1,y1,x2,y2), k)

            if best_candidate:
                last_roi = best_candidate[1]
                # smoothing iniziale
                last_k = best_candidate[2]
                k_hist.clear(); k_hist.append(last_k.copy())
                lost_cnt = 0
            else:
                lost_cnt += 1

        # --- se abbiamo ROI, aggiorna landmark + filtra ---
        if last_roi is not None:
            x1,y1,x2,y2 = last_roi
            x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
            if x2<=x1 or y2<=y1:
                last_roi=None; last_k=None; k_hist.clear()
            else:
                crop = frame[y1:y2, x1:x2]
                crop256 = cv2.resize(crop,(OUT_SIZE,OUT_SIZE),interpolation=cv2.INTER_LINEAR)
                flagp, k = run_landmark_dual(lm, lm_in, lm_out, crop256)

                # outlier check
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

                # draw
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                kk = last_k if last_k is not None else k
                for i in range(21):
                    xn,yn = float(kk[2*i]), float(kk[2*i+1])
                    x = int(x1 + xn*(x2-x1)); y = int(y1 + yn*(y2-y1))
                    cv2.circle(vis,(x,y),2,(0,255,255),-1)

                cv2.putText(vis, f"flag:{flagp:.2f} out:{out_frac*100:.0f}% lost:{lost_cnt}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        else:
            cv2.putText(vis,"No hand",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Step2 - Best-of-K + FaceReject", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_id += 1

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
