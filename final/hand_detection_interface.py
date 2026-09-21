#!/usr/bin/env python
# coding: utf-8

import cv2
import string
import numpy as np
from MediapipeModels import MediapipeHandModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


class GUI:
    def __init__(self, cls_path_name: str):
        self.init_mediapipe_components()
        self.classifier = self.load_classificator(cls_path_name)
        self.current_pred = ""
        self.current_phrase = ""
        self.info_text = "press 'space' to save char\npress 'd' to delete phrase\npress 'q' to quit"

    def init_mediapipe_components(self):
        """
        Inizializza i componenti di mediapipe
        """
        self.mp_class = MediapipeHandModel()
        self.mp_model = self.mp_class.return_hand_model()
        self.mp_hands = self.mp_class.return_mp_hands()
        self.mp_drawing = self.mp_class.return_mp_drawing()
        self.mp_drawing_styles = self.mp_class.return_mp_drawing_styles()

    def load_classificator(self, model_name: str):
        """
        Carica il classificatore keras (.h5 o .keras)
        """
        loaded_model = keras.models.load_model(model_name)
        return loaded_model

    def norm_min_max(self, vec: np.array) -> list:
        scaler = MinMaxScaler()
        scaler.fit(vec)
        norm = scaler.transform(vec)
        norm = norm.reshape(1, len(norm))[0]
        return list(norm)

    def put_info_text(self, image: np.array, text: str) -> np.array:
        x_pos = image.shape[0] - 50
        y_pos = 10
        for i, line in enumerate(text.split('\n')):
            cv2image = cv2.putText(
                img=image,
                text=line,
                org=(x_pos, y_pos),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.4,
                color=(58, 255, 255),
                thickness=1
            )
            y_pos += 12
        return cv2image

    def put_pred_text(self, image: np.array, text: str) -> np.array:
        return cv2.putText(
            img=image,
            text=text,
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.5,
            color=(24, 57, 255),
            thickness=3
        )

    def put_select_text(self, image: np.array, text: str) -> np.array:
        return cv2.putText(
            img=image,
            text=text,
            org=(20, 80),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.5,
            color=(125, 246, 55),
            thickness=3
        )

    def label_to_str(self, label: int) -> str:
        return str(list(string.ascii_lowercase)[label])

    def get_mediapipe_keypoints(self, landmark, window_w: int, window_h: int) -> list:
        """
        Estrae i keypoints di mediapipe come vettore non normalizzato
        """
        vector = []
        for markers in landmark:
            for mark in range(len(markers.landmark)):
                vector.append(markers.landmark[mark].x * window_w)
                vector.append(markers.landmark[mark].y * window_h)
        return vector

    def show_hand_keypoints(self, cv2image: np.array) -> np.array:
        """
        show_hand_keypoints()

        This method is used to print in cv2 image the hand keypoints and the text of predicted label.
        """

        # Converti da BGR a RGB prima di passarlo a Mediapipe
        results_img = self.mp_model.process(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))

        # Check if there are landmarks
        if results_img.multi_hand_landmarks:

            # Load keypoints to vectors
            vector = self.get_mediapipe_keypoints(
                results_img.multi_hand_landmarks,
                cv2image.shape[1],
                cv2image.shape[0]
            )

            # Normalize keypoint vector
            vector = self.norm_min_max(np.array(vector).reshape(-1, 1))

            # Check if there is only one hand detected (one hand have 42 features in this case)
            if len(vector) == 42:
                # Convert to np.array for keras input
                input_vec = np.array(vector).reshape(1, -1)

                # Predict with keras model
                pred_probs = self.classifier.predict(input_vec, verbose=0)
                pred_class = np.argmax(pred_probs, axis=1)[0]

                # Convert label pred (int) to alphabet letter (str)
                self.current_pred = self.label_to_str(int(pred_class))

                # Put text on cv2 preview
                self.put_pred_text(cv2image, self.current_pred)

            # Show keypoints
            for hand_landmarks in results_img.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    cv2image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return cv2image

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

                # 🔑 Leggi il tasto premuto UNA sola volta
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

        # Release camera
        cap.release()
        cv2.destroyAllWindows()
        print("Window closed.")

