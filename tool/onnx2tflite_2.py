import os
import cv2
import onnx
import numpy as np
import tensorflow as tf

import sys

sys.path.append(r"D:\algorithm\ultralytics\tool\onnx2tf")  # onnx2tflite的地址

from onnx2tf.onnx_2tf import ONNX2TF

if __name__ == '__main__':

    onnx_model_path = r'D:\algorithm\ultralytics\tflite\yolov8n_12.onnx' # 需要转换的onnx文件位置 
    tflite_out_path = r"D:\algorithm\ultralytics\tflite"

    save_path = ONNX2TF(onnx_model_path,tflite_out_path,is_quant= False,
                        is_int8= False,image_root=None,
                        # int8quant_mean= [94.964778,76.867057,66.213701],
                        # int8quant_std= [75.883071,68.674585,64.933873]
                        int8quant_mean=[127.5, 127.5, 127.5],
                        int8quant_std=[127.5, 127.5, 127.5])
                        
    print(save_path)