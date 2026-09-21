#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import pickle
import string
import numpy as np
from collections import deque
from MediapipeModels import MediapipeHandModel
from tensorflow import keras
import tensorflow as tf
import serial
import time


class GUI:
    def __init__(self, cls_path_name: str):
        self.init_mediapipe_components()
        self.classifier = self.load_classificator(cls_path_name)
        self.classifier_quant = tf.lite.Interpreter(model_path = "lis_dnn_model_quant_2.tflite")

        self.ser = serial.Serial(
            port="COM3",
            baudrate=115200,
            timeout=3
        )

        # Carica scaler e encoder (se presenti)
        base_dir = os.path.dirname(os.path.abspath(cls_path_name))
        self.encoder = self.load_label_encoder(os.path.join(base_dir, "label_encoder.pkl")) or self.load_label_encoder(
            "../final/label_encoder.pkl")
        self.scaler = self.load_scaler(os.path.join(base_dir, "scaler.pkl")) or self.load_scaler("../final/scaler.pkl")

        # Classi
        if self.encoder is not None:
            self.class_names = list(self.encoder.classes_)
        else:
            self.class_names = list(string.ascii_lowercase)

        # Parametri
        self.prob_buffer = deque(maxlen=7)
        self.conf_threshold = 0.6

        self.current_pred = ""
        self.current_phrase = ""
        self.info_text = "press 'space' to save char\npress 'd' to delete phrase\npress 'q' to quit"

    def init_mediapipe_components(self):
        self.mp_class = MediapipeHandModel()
        self.mp_model = self.mp_class.return_hand_model()
        self.mp_hands = self.mp_class.return_mp_hands()
        self.mp_drawing = self.mp_class.return_mp_drawing()
        self.mp_drawing_styles = self.mp_class.return_mp_drawing_styles()

    def load_classificator(self, model_name: str):
        return keras.models.load_model(model_name)

    def load_label_encoder(self, path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def load_scaler(self, path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def get_mediapipe_keypoints(self, landmark, window_w: int, window_h: int) -> list:
        vector = []
        for markers in landmark:
            for mark in range(len(markers.landmark)):
                vector.append(markers.landmark[mark].x * window_w)
                vector.append(markers.landmark[mark].y * window_h)
        return vector

    def put_info_text(self, image: np.array, text: str) -> np.array:
        x_pos = image.shape[0] - 50
        y_pos = 10
        for line in text.split('\n'):
            cv2.putText(
                image, line,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4, (58, 255, 255), 1
            )
            y_pos += 12
        return image

    def put_pred_text(self, image: np.array, text: str) -> np.array:
        return cv2.putText(
            image, text,
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5, (24, 57, 255), 3
        )

    def put_select_text(self, image: np.array, text: str) -> np.array:
        return cv2.putText(
            image, text,
            (20, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5, (125, 246, 55), 3
        )

    def put_topk(self, image: np.array, probs: np.array, k: int = 3) -> np.array:
        if probs is None:
            return image
        idx = np.argsort(probs)[-k:][::-1]
        y0 = 110
        for r, i in enumerate(idx):
            label = self.class_names[i] if i < len(self.class_names) else str(i)
            txt = f"{r+1}) {label}: {probs[i]:.2f}"
            cv2.putText(image, txt, (20, y0 + 20*r),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
        return image

    def show_hand_keypoints(self, cv2image: np.array) -> np.array:
        results_img = self.mp_model.process(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))

        if results_img.multi_hand_landmarks:
            vector = self.get_mediapipe_keypoints(
                results_img.multi_hand_landmarks,
                cv2image.shape[1],
                cv2image.shape[0]
            )

            if len(vector) == 42:
                x = np.array(vector, dtype=np.float32).reshape(1, -1)
                if self.scaler is not None:
                    x = self.scaler.transform(x)
                probs = self.perform_prediction(x)
                self.prob_buffer.append(probs)
                avg_probs = np.mean(self.prob_buffer, axis=0)

                pred_idx = int(np.argmax(avg_probs))
                conf = float(avg_probs[pred_idx])

                if self.encoder is not None:
                    label = self.encoder.inverse_transform([pred_idx])[0]
                else:
                    label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)

                self.current_pred = label if conf >= self.conf_threshold else "?"

                self.put_pred_text(cv2image, self.current_pred)
                self.put_topk(cv2image, avg_probs, k=3)

            for hand_landmarks in results_img.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    cv2image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return cv2image

    def perform_prediction(self, in_buffer):
        #probs = self.classifier.predict(in_buffer, verbose=0)[0]

        self.classifier_quant.allocate_tensors()
        input_details = self.classifier_quant.get_input_details()
        output_details = self.classifier_quant.get_output_details()

        print(input_details)
        print(output_details)

        in_scale, in_zero_point = input_details[0]["quantization"]
        in_buffer_int8 = (in_buffer/in_scale + in_zero_point).astype(np.int8)

        # Converte in stringa con ":" e termina con newline (\n)
        message = ":".join(str(x) for x in in_buffer_int8[0]) + "\n"
        print("Sending:", message)

        # --- Invia i dati ---
        self.ser.write(message.encode("utf-8"))

        # --- Delay per dare tempo alla board di elaborare ---
        time.sleep(0.1)

        response = self.ser.read_all().decode(errors='ignore').strip()

        probs = []
        if response:
            try:
                print("Received from STM32:")
                probs = [int(x) for x in response.split(':') if x]
                print(probs)
            except:
                probs = []
        else:
            print("Nessuna risposta ricevuta dalla STM32")


        self.classifier_quant.set_tensor(input_details[0]["index"], in_buffer_int8)
        self.classifier_quant.invoke()
        out_buffer = self.classifier_quant.get_tensor(output_details[0]["index"])


        probs = out_buffer[0]

        return probs

    def main_gui(self):
        width, height = 360, 360
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while True:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = self.show_hand_keypoints(frame)
                    self.put_select_text(frame, self.current_phrase)
                    self.put_info_text(frame, self.info_text)
                    cv2.imshow('Frame', frame)
                else:
                    print("Frame not captured.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    self.current_phrase += self.current_pred
                if key == ord('d'):
                    self.current_phrase = ""
                if key == ord('q'):
                    break
            else:
                print("Cannot open camera.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Window closed.")
