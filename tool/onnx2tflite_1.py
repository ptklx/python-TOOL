import sys

if True:
    # onnx
    # onnxruntime
    # onnx-simplifier
    # numpy
    # tensorflow>=2.5
    # opencv-python
    # https://github.com/MPolaris/onnx2tflite
    sys.path.append(r"D:\algorithm\ultralytics\tool\onnx2tflite")  # onnx2tflite的地址

    from converter import onnx_converter
    # onnx_path = r'D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify_test.onnx' # 需要转换的onnx文件位置
    onnx_path = r'D:\algorithm\yolov5\runs\train\exp7\weights\best.onnx' # 需要转换的onnx文件位置 
    out_path = r"D:\algorithm\yolov5\runs\train\exp7\weights"
    onnx_converter(
        onnx_model_path = onnx_path,
        need_simplify = True,
        output_path = out_path,  # 输出的tflite存储路径
        target_formats = ['tflite'], # or ['keras'], ['keras', 'tflite']
        weight_quant = False,
        int8_model = False,
        int8_mean = None,
        int8_std = None,
        image_root = None
    )

    print("ok!")
else:
    sys.path.append(r"D:\algorithm\yolov5\tool\onnx2tf")  # onnx2tflite的地址

    from onnx2tf.onnx2tflite import ONNX2TFLiteConvertor
    # onnx 版本不能超过12
    onnx_path = r'D:\algorithm\yolov5\tflite\yolov5n.onnx' # 需要转换的onnx文件位置 
    out_path = r"D:\algorithm\yolov5\tflite"
    ONNX2TFLiteConvertor(onnx_path,
                        out_path,
                        None,
                        need_simplify=False,
                        is_quant=False,
                        is_int8=False,
                        # image_root="/data/zhangxian/data/coco/images/val2017/",
                        image_root=r"D:\algorithm\datasets\coco128\images\train2017",
                    #  int8quant_mean=None,
                    #  int8quant_std=None)
                    #  int8quant_mean=[123.675, 116.28, 103.53],
                    #  int8quant_std=[58.395, 57.12, 57.375])
                        int8quant_mean=[127.5, 127.5, 127.5],
                        int8quant_std=[127.5, 127.5, 127.5])
    print("ok!")