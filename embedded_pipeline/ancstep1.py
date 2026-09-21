# step1_palm_crop256_centered_pos2.py
import cv2
import numpy as np
import tensorflow as tf

PALM_TFLITE = "models/palm_detection.tflite"

INPUT_RANGE = "zero_one"   # se non rileva, prova "neg1_1"
SCORE_TH    = 0.35
NMS_IOU_TH  = 0.30
SCALE       = 1.35         # margine del crop quadrato
OUT_SIZE    = 256

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

# ---------- Decode + NMS ----------
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
    keep=[]
    while idx.size and len(keep) < topk:
        i = idx[0]; keep.append(i)
        if idx.size == 1: break
        idx = idx[1:][ iou_batch(xyxy[i],xyxy[idx[1:]]) < iou_th ]
    return keep

def decode_palm_with_keypoints_pos2(classificators, regressors, anchors, score_th=SCORE_TH):
    """
    POS_VARIANT = 2:
      xc = axc + dx/256       (NO *aw/ah)
      yc = ayc + dy/256
      kx = axc + kx_off/256   (stesso per i 7 keypoint)
      ky = ayc + ky_off/256
    SIZE: exp(log) come prima.
    """
    # atteso: [1,N,1] e [1,N,18]; se invertiti, scambiali
    if classificators.shape[-1] == 18 and regressors.shape[-1] == 1:
        classificators, regressors = regressors, classificators

    scores = sigmoid_stable(classificators[0, :, 0])   # logit -> prob
    reg = regressors[0, :, :]                          # [N,18]

    dets   = []
    all_kp = []

    for i, (axc, ayc, aw, ah) in enumerate(anchors):
        s = scores[i]
        if s < score_th:
            continue

        dx, dy, dw, dh = reg[i, :4]

        # --- POS VARIANT 2 (no *aw/ah) ---
        xc = axc + (dx / 256.0)
        yc = ayc + (dy / 256.0)

        # --- SIZE exp su /256 (log-scale) ---
        w  = np.exp(np.clip(dw, -10.0, 10.0) / 256.0) * aw
        h  = np.exp(np.clip(dh, -10.0, 10.0) / 256.0) * ah

        # --- 7 keypoint (stesso pos2) ---
        kps = []
        for j in range(7):
            kx_off = reg[i, 4 + 2*j]
            ky_off = reg[i, 4 + 2*j + 1]
            kx = axc + (kx_off / 256.0)
            ky = ayc + (ky_off / 256.0)
            kps.append([kx, ky])
        kps = np.array(kps, dtype=np.float32)

        dets.append([s, xc, yc, w, h])
        all_kp.append(kps)

    if not dets:
        return np.empty((0,5), np.float32), np.empty((0,7,2), np.float32)

    dets = np.array(dets, np.float32)
    all_kp = np.stack(all_kp, axis=0)  # [K,7,2]

    # NMS
    xyxy = np.stack([dets[:,1]-dets[:,3]/2, dets[:,2]-dets[:,4]/2,
                     dets[:,1]+dets[:,3]/2, dets[:,2]+dets[:,4]/2], 1)
    keep = nms(xyxy, dets[:,0], NMS_IOU_TH, topk=5)
    return dets[keep], all_kp[keep]

# ---------- Crop centrato sui keypoint ----------
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

# ---------- MAIN ----------
def main():
    interp = tf.lite.Interpreter(model_path=PALM_TFLITE)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    cap = cv2.VideoCapture(0)
    print("Premi 'q' per uscire.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im  = cv2.resize(rgb, (256,256)).astype(np.float32) / 255.0
        if INPUT_RANGE == "neg1_1":
            im = im*2.0 - 1.0

        interp.set_tensor(inp[0]['index'], im[None, ...])
        interp.invoke()
        cls = interp.get_tensor(out[0]['index'])
        reg = interp.get_tensor(out[1]['index'])

        dets, kps = decode_palm_with_keypoints_pos2(cls, reg, ANCHORS)

        vis = frame.copy()
        if dets.shape[0]:
            best_idx = int(np.argmax(dets[:,0]))
            best_det = dets[best_idx]
            best_kps = kps[best_idx]

            crop256, (x1,y1,x2,y2) = crop_centered(vis, best_det, best_kps, SCALE, OUT_SIZE)

            # draw
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{best_det[0]:.2f}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            for (kx,ky) in best_kps:
                cv2.circle(vis, (int(kx*vis.shape[1]), int(ky*vis.shape[0])), 3, (0,255,255), -1)

            if crop256 is not None:
                cv2.imshow("Crop 256x256", crop256)
        else:
            cv2.putText(vis, "No hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            try: cv2.destroyWindow("Crop 256x256")
            except: pass

        cv2.imshow("Palm step1 (centered pos2)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
