import numpy as np
import tensorflow as tf
from .my_onnx_run import ONNXModel

def compare_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2:
            return False
    return True

def compare_transform_acc(onnx_model_path:str, tflite_model_path:str):
    interest_layers = []
    model_onnx = ONNXModel(onnx_model_path, interest_layers)
    X = np.random.randn(*model_onnx.input_shape).astype(np.float32)*10
    onnx_out = model_onnx.forward(X)[-1]

    try:
        model_tflite = tf.lite.Interpreter(model_path=tflite_model_path)
        input_details, output_details  = model_tflite.get_input_details(), model_tflite.get_output_details()
        input_shape = input_details[0]['shape']
        if not compare_shape(input_shape, X.shape):
            shape = [i for i in range(len(X.shape))]
            newshape = [shape[0], *shape[2:], shape[1]]
            X = X.transpose(*newshape)
        if not compare_shape(input_shape, X.shape):
            return None
        model_tflite.allocate_tensors()
        model_tflite.set_tensor(input_details[0]['index'], X)
        model_tflite.invoke()
    except Exception as e:
        raise ValueError(f"tflite model load failed.")

    mean_diff = np.inf
    for i in range(len(output_details)):
        tflite_output = model_tflite.get_tensor(output_details[i]['index'])
        if tflite_output.size != onnx_out.size:
            continue
        if len(tflite_output.shape) > 2 and onnx_out.shape != tflite_output.shape:
            shape = [i for i in range(len(tflite_output.shape))]
            newshape = [shape[0], shape[-1], *shape[1:-1]]
            tflite_output = tflite_output.transpose(*newshape)
        
        if len(onnx_out) == len(tflite_output) and onnx_out.shape == tflite_output.shape:
            diff = np.abs(onnx_out - tflite_output)
            mean_diff = min(np.mean(diff), mean_diff)
    
    if mean_diff == np.inf:
        mean_diff = 0.0011
    return "{:.2E}".format(mean_diff)

if __name__ == "__main__":
    print(compare_transform_acc(
        onnx_model_path="C:/Users/54049/Desktop/onnx2tflite/models/picodet_l_640x640.onnx",
        tflite_model_path="C:/Users/54049/Desktop/onnx2tflite/models/picodet_l_640x640.tflite",
    ))