import os
import cv2
import onnx
import onnx_tf
import numpy as np
import tensorflow as tf

# from PATH_CONFIG import *
import sys
if sys.platform.startswith('win'):
    print('This is Windows platform.')
    ONNXTF_TEMP_FLODER = "./"
elif sys.platform.startswith('linux'):
    ONNXTF_TEMP_FLODER = "/home/onnxtf_convert_temp"


def representative_dataset_gen(img_root, batch_size, img_size, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    if isinstance(mean, list):
        mean = np.array(mean, dtype=np.float32)
    if isinstance(std, list):
        std = np.array(std, dtype=np.float32)
        std = np.maximum(std, np.ones_like(std)*0.0001)

    if img_root is None or (not os.path.exists(img_root)):
        raise FileNotFoundError(f"tflite int8 quant needs datas")

    VALID_FORMAT = ['jpg', 'png', 'jpeg', 'bmp']
    for i, fn in enumerate(os.listdir(img_root)):
        if fn.split(".")[-1].lower() not in VALID_FORMAT:
            continue
        _input = cv2.imread(os.path.join(img_root, fn))
        if _input is None:
            continue
        _input = cv2.resize(_input, (img_size[1], img_size[0]))[:, :, ::-1]
        if mean is not None:
            _input = (_input - mean)
        if std is not None:
            _input = _input/std

        _input = _input[None, ...]
        _input = _input.astype(np.float32)
        _input = np.repeat(_input, batch_size, axis=0)
        if img_size[1] < 5:
            _input = _input.transpose(0, 3, 1, 2)
        yield [_input]
        if i >= 100:
            break

def ONNX2TF(onnx_model_path:str, 
            tflite_out_path:str=None, 
            is_quant:bool=False, 
            is_int8:bool=False, 
            image_root:str=None,
            int8quant_mean:list or float=[123.675, 116.28, 103.53],
            int8quant_std:list or float=[58.395, 57.12, 57.375]):
    '''
        :param onnx_model_path: ONNX模型路径
        :param tflite_out_path: tflite模型输出路径, None为ONNX模型目录
        :param is_quant: 网络权重是否量化, 仅模型权重int8量化
        :param is_int8: 输入输出是否量化, 连同输入输出一起量化, 需要representative_dataset
        :param image_root: 输入输出int8量化需要, 用于计算zero,scale, 如果没有则随机产生数据
        :param int8quant_mean: int8量化图片预处理的mean
        :param int8quant_std: int8量化图片预处理的std
    '''
    onnx_model = onnx.load(onnx_model_path)
    input_shape = []
    for inp in onnx_model.graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        batch_size = max(input_shape[0], 1)
        break
    model = onnx_tf.backend.prepare(onnx_model)
    model.export_graph(ONNXTF_TEMP_FLODER)
    del onnx_model
    
    converter = tf.lite.TFLiteConverter.from_saved_model(ONNXTF_TEMP_FLODER)
    # converter.experimental_new_converter = True
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    if is_quant or is_int8:
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if is_int8:
        converter.representative_dataset = lambda: representative_dataset_gen(image_root, batch_size, (input_shape[2], input_shape[3]), int8quant_mean, int8quant_std)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    
    if tflite_out_path is None:
        tflite_out_path = os.path.split(onnx_model_path)[0]
    if not os.path.exists(tflite_out_path):
        os.makedirs(tflite_out_path)

    tflite_model_name = os.path.split(onnx_model_path)[-1]
    if tflite_model_name.endswith(".onnx"):
        tflite_model_name = tflite_model_name[:-5]
    if not tflite_model_name.endswith(".tflite"):
        if is_int8:
            tflite_model_name = tflite_model_name + "_int8.tflite"
        else:
            tflite_model_name = tflite_model_name + "_fp32.tflite"
    
    save_path = os.path.join(tflite_out_path, tflite_model_name)
    with open(save_path, "wb") as fp:
        fp.write(tflite_model)
    return save_path


if __name__ == '__main__':
    onnx_model_path = "/home/test/fire221012_640_38K.onnx"
    tflite_out_path="/home/test"
    save_path = ONNX2TF(onnx_model_path,tflite_out_path,is_quant= False,is_int8= False,image_root=None,int8quant_mean= [94.964778,76.867057,66.213701],int8quant_std= [75.883071,68.674585,64.933873])
    print(save_path)