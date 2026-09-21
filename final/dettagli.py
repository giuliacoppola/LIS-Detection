import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="../final_project/lis_dnn_model_quant_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== INPUT DETAILS ===")
print(input_details)
print("\n=== OUTPUT DETAILS ===")
print(output_details)
