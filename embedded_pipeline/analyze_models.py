import tensorflow as tf

model_paths = {
    "lis_classifier": "models/lis_dnn_model_quant_int8.tflite",
    "palm_detector": "models/palm_detection.tflite",
    "hand_landmark": "models/hand_landmark.tflite"
}

def print_model_info(name, path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n==== {name} ====")
    print("Input:")
    for inp in input_details:
        print(f" - name: {inp['name']}, shape: {inp['shape']}, dtype: {inp['dtype']}")
    print("Output:")
    for out in output_details:
        print(f" - name: {out['name']}, shape: {out['shape']}, dtype: {out['dtype']}")

for name, path in model_paths.items():
    print_model_info(name, path)
