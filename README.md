# LIS-Detection

Real-time recognition of the Italian Sign Language (LIS – *Lingua Italiana dei Segni*) fingerspelling alphabet, from hand-landmark detection all the way to on-device inference on a microcontroller with a dedicated NPU.

This repository started as a fork of [xandrew94x/LIS-Detection](https://github.com/xandrew94x/LIS-Detection) (© Andrea Sciuto, GNU GPLv3), which implements hand detection with MediaPipe + OpenCV and a K-NN classifier. For my bachelor's thesis in Computer Engineering at the Università di Napoli Federico II, I extended that baseline into a full pipeline: a trained neural-network classifier, INT8 quantization, a MediaPipe-free landmark pipeline built directly on the raw TFLite models, and deployment on an STM32N6 microcontroller running the network on its integrated NPU.

<img src="readmeFiles/lis_example.gif" />

## Credits & license

- **Base project (hand detection + K-NN classifier):** [xandrew94x/LIS-Detection](https://github.com/xandrew94x/LIS-Detection) — © Andrea Sciuto.
- **Extensions (DNN classifier, quantization, MediaPipe-free pipeline, STM32/NPU deployment):** Giulia Coppola, 2025, bachelor thesis work.
- Released under the same **GNU General Public License v3.0** as the original project (see [LICENSE](LICENSE)).

## Pipeline overview

```
Webcam frame
   │
   ▼
Hand detection + 21 landmarks (x, y)   → MediaPipe (desktop demos) or
   │                                     raw BlazePalm/HandLandmark TFLite models (embedded_pipeline/)
   ▼
42-feature vector (right hand only, MinMax-normalized)
   │
   ▼
Classifier: K-NN (baseline) → Dense NN (128-64-26, this work)
   │
   ▼
INT8 quantization (TFLite)
   │
   ▼
 ┌────────────────────────────┬──────────────────────────────────┐
 │ Desktop demo (webcam, GUI)  │ On-device inference (STM32N6 NPU) │
 └────────────────────────────┴──────────────────────────────────┘
```

## Repository layout

| Path | Description |
|---|---|
| `py/` | Original baseline demo (MediaPipe + K-NN), from the upstream repo. |
| `classifier/knn_model.pkl` | Original K-NN model. |
| `handDataset2/` | LIS alphabet dataset (26 classes) acquired for this thesis. |
| `train.py`, `train_model.py` | Training of the Dense NN classifier that replaces the K-NN baseline. |
| `quant.py`, `calib_data.py`, `check_dataset.py`, `fix_invalid_vector.py`, `test_quant.py`, `errore h5_tflite.py` | Keras → TFLite INT8 quantization pipeline, plus calibration data and the conversion issues encountered along the way. |
| `evaluate_classifier.py`, `matrice.py`, `grafico.py` | Evaluation utilities (accuracy, confusion matrix, plots). |
| `final/` | Intermediate iteration: DNN-based desktop demo + evaluation/comparison scripts (confusion matrix, accuracy comparison, flow diagram). |
| `final_project/` | Final desktop demo app (webcam + MediaPipe + quantized DNN classifier). |
| `embedded_pipeline/` | Hand-landmark pipeline reimplemented on the **raw TFLite models** (BlazePalm anchor decoding, NMS, hand-landmark model) with no MediaPipe dependency, used to validate the exact numerical pipeline that runs on the microcontroller. Includes `uart_test.py` to exchange characters with the board over serial. |
| `stm32/progetto_tesi2/` | STM32CubeIDE project (NUCLEO-N657X0-Q, STM32N6, secure boot FSBL + Appli, X-CUBE-AI 10.2.0) that runs the quantized network on the board's Neural-ART NPU. |
| `ipynb/` | Notes on classification and data normalization. |

## :hammer: Technologies

- Python 3.8 / 3.12
- OpenCV, MediaPipe Hand Landmarker
- scikit-learn (baseline K-NN, preprocessing)
- TensorFlow / Keras (Dense NN classifier) + TFLite (INT8 quantization)
- pyserial (UART bridge to the board)
- STM32CubeIDE, X-CUBE-AI 10.2.0, NUCLEO-N657X0-Q (STM32N6 + Neural-ART NPU)

## :books: Install

```
pip install -r pip/requirements.txt
```

`pip/requirements.txt` covers the desktop pipeline (OpenCV, MediaPipe, scikit-learn, TensorFlow). `embedded_pipeline/uart_test.py` additionally needs `pyserial`.

## :bar_chart: Dataset

- Original K-NN baseline dataset (26 LIS letters, 500 vectors/class, right hand only): acquired with [acqTool](https://github.com/xandrew94x/acqTool), downloadable [here](https://www.kaggle.com/datasets/andrewk94/handrightdataset).
- `handDataset2/` in this repo: the dataset acquired for this thesis to train the Dense NN classifier, same 42-feature format (`(x1, y1, ..., x21, y21)`, MinMax-normalized in `[0, 1]`).

## :rocket: Running it

**Baseline desktop demo (K-NN, original repo):**
```
python py/hand_detection_main.py -c classifier/knn_model.pkl
```

**Final desktop demo (Dense NN classifier):**
```
python final_project/hand_detection_main_final.py -c final_project/lis_dnn_model.keras
```

**Train the Dense NN classifier on `handDataset2/`:**
```
python train_model.py
```

**Quantize the trained model to INT8 TFLite:**
```
python quant.py
```

**Validate the MediaPipe-free pipeline (raw TFLite models, same numerics as the embedded target):**
```
python embedded_pipeline/run_pipeline.py
```

**Talk to the STM32 board over UART:**
```
python embedded_pipeline/uart_test.py
```

**Flash the STM32 project:** open `stm32/progetto_tesi2/progetto_tesi2.ioc` in STM32CubeIDE (FSBL + Appli, X-CUBE-AI 10.2.0) and build/flash the NUCLEO-N657X0-Q board. Build outputs (`Debug/`, `.bin`, `.elf`, ...) are not versioned; the source, linker scripts and the X-CUBE-AI generated network are.

## :children_crossing: Classifier

Confusion matrix of the K-NN baseline:

![confusion_matrix_knn](readmeFiles/confusion_matrix_knn.png)

## :page_facing_up: License

This project is released under the [GNU General Public License v3.0](LICENSE), the same license as the upstream project it builds on.
