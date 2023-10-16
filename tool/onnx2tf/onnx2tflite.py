'''
    ONNX到tflite转换工具
    author: 肖禾
    time: 2022/01/07
'''
import os
import cv2
import onnx
import numpy as np
import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
import sys
sys.path.append(r"D:\algorithm\ultralytics\tool\onnx2tf")
# from onnx2tf import operators
import operators

try:
    from onnxsim import simplify
except:
    print("引入onnxsim.simplify失败")
    def lambda_func(x, *arg, **args):
        return x, False
    simplify = lambda_func

def representative_dataset_gen(img_root, batch_size, img_size, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],color_model='rgb'):
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
        if color_model=='rgb' or color_model is None:
            _input = cv2.resize(_input, (img_size[1], img_size[0]))[:, :, ::-1]
        else:
            _input = cv2.resize(_input, (img_size[1], img_size[0]))
        if mean is not None:
            _input = (_input - mean)
        if std is not None:
            _input = _input/std

        _input = _input[None, ...]
        _input = _input.astype(np.float32)
        _input = np.repeat(_input, batch_size, axis=0)
        yield [_input]
        if i >= 100:
            break

def ONNX2TFLiteConvertor(onnx_model_path:str, 
                            tflite_out_path:str=None, 
                            tflite_model_name:str=None, 
                            need_simplify:bool=True, 
                            is_quant:bool=False, 
                            is_int8:bool=False, 
                            image_root:str=None,
                            int8quant_mean:list or float=[123.675, 116.28, 103.53],
                            int8quant_std:list or float=[58.395, 57.12, 57.375]):
    '''
        :param onnx_model_path: ONNX模型路径
        :param tflite_out_path: tflite模型输出路径, None为ONNX模型目录
        :param tflite_model_name: 输出模型名称,None为ONNX模型名称
        :param need_simplify: 模型结构是否需要优化, 依赖onnxsim.simplify
        :param is_quant: 网络权重是否量化, 仅模型权重int8量化
        :param is_int8: 输入输出是否量化, 连同输入输出一起量化, 需要representative_dataset
        :param image_root: 输入输出int8量化需要, 用于计算zero,scale, 如果没有则随机产生数据
        :param int8quant_mean: int8量化图片预处理的mean
        :param int8quant_std: int8量化图片预处理的std
    '''
    model_proto = onnx.load(onnx_model_path)
    dynamic_input = False
    tf_tensor, input_shape = {}, []
    for inp in model_proto.graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape[0] == 0:
            dynamic_input = True
        batch_size = max(input_shape[0], 1)
        tf_tensor[inp.name] = keras.Input(shape=(input_shape[2], input_shape[3], input_shape[1]), batch_size=batch_size, name=inp.name)

    if need_simplify:
        success = False
        try:
            model_proto, success = simplify(model_proto, check_n=2, dynamic_input_shape=dynamic_input)
        except:
            success = False
        if not success:
            model_proto = onnx.load(onnx_model_path)

    model_graph = model_proto.graph
    onnx_weights = {}
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)
    
    for node in model_graph.node:
        op_name, node_inputs, node_outputs = node.op_type, node.input, node.output
        op_attr = {}
        for x in node.attribute:
            if x.type == 1:
                op_attr[x.name] = x.f
            elif x.type == 2:
                op_attr[x.name] = x.i
            elif x.type == 3:
                op_attr[x.name] = x.s.decode()
            elif x.type == 7:
                op_attr[x.name] = x.ints
        if "Conv" == op_name:
            c2, c1 = onnx_weights[node_inputs[1]].shape[:2]
            weights = onnx_weights[node_inputs[1]].transpose(2,3,1,0)
            dilations, group = op_attr['dilations'], op_attr['group']
            if "pads" in  op_attr:
                pads = op_attr['pads']
            else:
                pads = None
            kernel_shape, strides = op_attr['kernel_shape'], op_attr['strides']
            bias = onnx_weights[node_inputs[2]] if len(node_inputs) == 3 else None

            if group == 1:
                tf_tensor[node_outputs[0]] = operators.TFConv(c1,c2,kernel_shape,strides, dilations, pads,group,weights,bias)(tf_tensor[node_inputs[0]])
            elif group == c2:
                weights = weights.transpose(0, 1, 3, 2)
                tf_tensor[node_outputs[0]] = operators.TFDepthwiseConv2D(kernel_shape, strides, dilations, pads, weights, bias)(tf_tensor[node_inputs[0]])
            else:
                tf_tensor[node_outputs[0]] = operators.TFGroupConv(c1,c2,kernel_shape,strides, dilations, pads,group,weights,bias)(tf_tensor[node_inputs[0]])
        elif "ConvTranspose" == op_name:
            c2, c1 = onnx_weights[node_inputs[1]].shape[:2]
            weights = onnx_weights[node_inputs[1]].transpose(2,3,1,0)
            dilations, group = op_attr['dilations'], op_attr['group']
            if "pads" in  op_attr:
                pads = op_attr['pads']
            else:
                pads = None
            kernel_shape, strides = op_attr['kernel_shape'], op_attr['strides']
            bias = onnx_weights[node_inputs[2]] if len(node_inputs) == 3 else None
            tf_tensor[node_outputs[0]] = operators.TFConvTranspose(c1,c2,kernel_shape,strides, dilations, pads,group,weights,bias)(tf_tensor[node_inputs[0]])
        elif "BatchNormalization" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFBatchNormalization(
                weight = onnx_weights[node_inputs[1]],
                bias = onnx_weights[node_inputs[2]],
                running_mean = onnx_weights[node_inputs[3]],
                running_var = onnx_weights[node_inputs[4]],
                epsilon = op_attr.get("epsilon"),
                momentum = op_attr.get("momentum")
                )(tf_tensor[node_inputs[0]])
        elif "InstanceNormalization" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFInstanceNormalization(
                scale = onnx_weights[node_inputs[1]],
                bias = onnx_weights[node_inputs[2]],
                epsilon = op_attr.get("epsilon")
                )(tf_tensor[node_inputs[0]])
        elif "Relu" == op_name:
            tf_tensor[node_outputs[0]] = keras.activations.relu(tf_tensor[node_inputs[0]])
        elif "Sigmoid" == op_name:
            tf_tensor[node_outputs[0]] = keras.activations.sigmoid(tf_tensor[node_inputs[0]])
        elif "LeakyRelu" == op_name:
            tf_tensor[node_outputs[0]] = keras.activations.relu(tf_tensor[node_inputs[0]], alpha=op_attr['alpha'])
        elif "Tanh" == op_name:
            tf_tensor[node_outputs[0]] = keras.activations.tanh(tf_tensor[node_inputs[0]])
        elif "PRelu" == op_name:
            slope = onnx_weights[node_inputs[1]].transpose(1, 2, 0)
            tf_tensor[node_outputs[0]] = tf.keras.layers.PReLU(weights=[slope], shared_axes = [1, 2])(tf_tensor[node_inputs[0]])
        elif "Selu" == op_name:
            tf_tensor[node_outputs[0]] = keras.activations.selu(tf_tensor[node_inputs[0]])
        elif "Softmax" == op_name:
            axis = operators.Torch2TFAxis(op_attr.get('axis', 1))
            tf_tensor[node_outputs[0]] = keras.activations.softmax(tf_tensor[node_inputs[0]], axis=axis)
        elif "AveragePool" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFAveragePool(op_attr.get("kernel_shape", 2)[0], op_attr.get("pads", None), op_attr.get("strides", 1)[0])(tf_tensor[node_inputs[0]])
        elif "MaxPool" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFMaxPool(op_attr.get("kernel_shape", 2)[0], op_attr.get("pads", None), op_attr.get("strides", 1)[0])(tf_tensor[node_inputs[0]])
        elif "Add" == op_name:
            t1 = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[0]])
            t2 = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            t1 = operators.convertType(t1)
            t2 = operators.convertType(t2)
            if len(t1.shape) == 4 and len(t2.shape) == 3:
                t2 = np.transpose(t2, [1, 2, 0])
            # yolov6
            if isinstance(t1, (int, float, np.ndarray)):
                t1, t2 = t2, t1
            if (len(t1.shape) == 2 or len(t1.shape) == 3) and isinstance(t2, np.ndarray) and  t2.ndim == 2:
                t2 = t2.transpose(1, 0)
            tf_tensor[node_outputs[0]] = t1 + t2
        elif "Sub" == op_name:
            t1 = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[0]])
            t2 = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            t1 = operators.convertType(t1)
            t2 = operators.convertType(t2)
            if len(t1.shape) == 4 and len(t2.shape) == 3:
                t2 = np.transpose(t2, [1, 2, 0])
            # # yolov6
            # if isinstance(t1, (int, float, np.ndarray)):
            #     t1, t2 = t2, t1
            if (len(t1.shape) == 2 or len(t1.shape) == 3) and isinstance(t2, np.ndarray) and  t2.ndim == 2:
                t2 = t2.transpose(1, 0)
            tf_tensor[node_outputs[0]] = t1 - t2
        elif "Mul" == op_name:    
            t1 = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[0]])
            t2 = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            t1 = operators.convertType(t1)
            t2 = operators.convertType(t2)
            if len(t1.shape) == 4 and len(t2.shape) == 3:
                t2 = np.transpose(t2, [1, 2, 0])
            # yolov6
            if isinstance(t1, (int, float, np.ndarray)):
                t1, t2 = t2, t1
            if (len(t1.shape) == 2 or len(t1.shape) == 3) and isinstance(t2, np.ndarray) and  t2.ndim == 2:
                t2 = t2.transpose(1, 0)
            tf_tensor[node_outputs[0]] = t1 * t2
        elif "Div" == op_name:
            t1 = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[0]])
            t2 = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            t1 = operators.convertType(t1)
            t2 = operators.convertType(t2)
            if len(t1.shape) == 4 and len(t2.shape) == 3:
                t2 = np.transpose(t2, [1, 2, 0])
            # # yolov6
            # if isinstance(t1, (int, float, np.ndarray)):
            #     t1, t2 = t2, t1
            if (len(t1.shape) == 2 or len(t1.shape) == 3) and isinstance(t2, np.ndarray) and  t2.ndim == 2:
                t2 = t2.transpose(1, 0)
            tf_tensor[node_outputs[0]] = t1 / t2
        elif "MatMul" == op_name:
            t1 = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[0]])
            t2 = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            tf_tensor[node_outputs[0]] = tf.matmul(t1, t2)
        elif "Concat" == op_name:
            gather = []
            for x in node_inputs:
                if x in tf_tensor:
                    gather.append(tf_tensor[x])
                else:
                    gather.append(operators.convertType(operators.TorchWeights2TF(onnx_weights[x])))
            axis = node.attribute[0].i
            if axis < 0:
                axis = len(tf_tensor[node_inputs[0]].shape) + axis
            axis = operators.Torch2TFAxis(axis)
            tf_tensor[node_outputs[0]] = tf.concat(gather, axis=axis)
        elif "Upsample" == op_name:
            _, h, w, _ = tf_tensor[node_inputs[0]].shape
            scale = onnx_weights[node_inputs[1]]
            if op_attr['mode'] == 'nearest':
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            else:
                method = tf.image.ResizeMethod.BILINEAR
            tf_tensor[node_outputs[0]] = tf.image.resize(tf_tensor[node_inputs[0]], (int(h*scale[2]), int(w*scale[3])), method=method)
        elif "Resize" == op_name:
            if len(node_inputs) == 4:
                # 从sizes取
                _, _, nh, nw = onnx_weights[node_inputs[3]]
            else:
                # 从scales取
                _, _, nh, nw = onnx_weights[node_inputs[-1]]
                _, h, w, _ = tf_tensor[node_inputs[0]].shape
                nh, nw = int(h*nh), int(w*nw)
            if op_attr['mode'] == 'nearest':
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            else:
                method = tf.image.ResizeMethod.BILINEAR
            tf_tensor[node_outputs[0]] = tf.image.resize(tf_tensor[node_inputs[0]], (nh, nw), method=method)
        elif "Constant" == op_name:
            val = numpy_helper.to_array(node.attribute[0].t)
            if(val.size == 0):
                val = np.array([0])
            tf_tensor[node_outputs[0]] = val
        elif "Slice" == op_name:
            data = tf_tensor[node_inputs[0]]
            shape = data.shape
            if len(node_inputs)==1:
                starts = op_attr['starts'][0]
                ends = op_attr['ends'][0]
                axis = operators.Torch2TFAxis(op_attr['axes'][0])
                steps = 1
            else:
                starts = onnx_weights[node_inputs[1]][0]
                ends = onnx_weights[node_inputs[2]][0]
                axis = operators.Torch2TFAxis(onnx_weights[node_inputs[3]][0])
                if ends> shape[axis]:
                    ends = shape[axis]
                steps = 1 if len(node_inputs) < 5 else onnx_weights[node_inputs[4]][0]
                tf_tensor[node_outputs[0]] = operators.TFSlice()(data, starts, ends, axis, steps)
            tf_tensor[node_outputs[0]] = operators.TFSlice()(data, starts, ends, axis, steps)
        elif "ScatterND" == op_name:
            axes_num = len(tf_tensor[node_inputs[0]].shape)
            trans_in = [0, axes_num-1] + [n for n in range(1, axes_num-1)]
            trans_out = [0] + [n for n in range(2, axes_num)] + [1]

            indices = tf.transpose(tf_tensor[node_inputs[1]], perm=trans_in) if node_inputs[1] in tf_tensor else onnx_weights[node_inputs[1]]
            updates = tf.transpose(tf_tensor[node_inputs[2]], perm=trans_in) if node_inputs[2] in tf_tensor else onnx_weights[node_inputs[2]]
            inputs = tf.transpose(tf_tensor[node_inputs[0]], perm=trans_in)
            tf_tensor[node_outputs[0]] = operators.TFScatterND()(inputs, indices, updates)
            tf_tensor[node_outputs[0]] = tf.transpose(tf_tensor[node_outputs[0]], perm=trans_out)
        elif "Shape" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFShape()(tf_tensor[node_inputs[0]])
        elif "Gather" == op_name:
            axis = operators.Torch2TFAxis(op_attr['axis'])
            indexs = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else onnx_weights[node_inputs[1]]
            tf_tensor[node_outputs[0]] = operators.TFGather()(tf_tensor[node_inputs[0]], indexs, axis)
        elif "ArgMax" == op_name:
            axis = operators.Torch2TFAxis(op_attr['axis'])
            keepdims = operators.Torch2TFAxis(op_attr.get('keepdims', 0))
            keepdims = keepdims != 0
            if keepdims:
                temp = tf.argmax(tf_tensor[node_inputs[0]], axis=axis)
                axis = len(tf_tensor[node_inputs[0]].shape) + axis if axis < 0 else axis
                tf_tensor[node_outputs[0]] = tf.expand_dims(temp, axis=axis)
            else:
                tf_tensor[node_outputs[0]] = tf.argmax(tf_tensor[node_inputs[0]], axis=axis)
        elif "Cast" == op_name:
            if op_attr['to'] == 1:
                tf_tensor[node_outputs[0]] = tf.cast(tf_tensor[node_inputs[0]], dtype=tf.float32)
            else:
                tf_tensor[node_outputs[0]] = tf.cast(tf_tensor[node_inputs[0]], dtype=tf.int32)
        elif "Floor" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFFloor()(tf_tensor[node_inputs[0]])
        elif "Unsqueeze" == op_name:
            axis = operators.Torch2TFAxis(op_attr['axes'][0])
            tf_tensor[node_outputs[0]] = operators.TFUnsequeeze()(tf_tensor[node_inputs[0]], axis)
        elif "Squeeze" == op_name:
            axis = op_attr['axes'] if "axes" in op_attr else onnx_weights[node_inputs[1]]
            axis = [operators.Torch2TFAxis(i) for i in axis]
            tf_tensor[node_outputs[0]] = tf.squeeze(tf_tensor[node_inputs[0]], axis=axis)
        elif "GlobalAveragePool" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFGlobalAveragePool()(tf_tensor[node_inputs[0]])
        elif "GlobalMaxPool" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFGlobalMaxPool()(tf_tensor[node_inputs[0]])
        elif "Flatten" == op_name:
            tf_tensor[node_outputs[0]] = operators.TFFlatten()(tf_tensor[node_inputs[0]])
        elif "Gemm" == op_name:
            # 全连接层
            weights = onnx_weights[node_inputs[1]]
            bias = onnx_weights[node_inputs[2]] if len(node_inputs) > 2 else None
            tf_tensor[node_outputs[0]] = operators.TFGemm(weights, bias)(tf_tensor[node_inputs[0]])
        elif "Pad" == op_name:
            if op_attr.get("pads"):
                pad = np.max(op_attr['pads'])
            elif node_inputs[1] in onnx_weights:
                # pad = np.max(onnx_weights[node_inputs[1]])
                # 来自 https://github.com/gmalivenko/onnx2keras 的魔法函数
                pads = onnx_weights[node_inputs[1]]
                pad = [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]]
                tf_tensor[node_outputs[0]] = keras.layers.ZeroPadding2D(
                    padding=((pads[2], pads[6]), (pads[3], pads[7]))
                )(tf_tensor[node_inputs[0]])
                continue
            else:
                pad = np.max(tf_tensor[node_inputs[1]])
            tf_tensor[node_outputs[0]] = operators.TFPad(pad, model=op_attr['mode'].upper())(tf_tensor[node_inputs[0]])
        elif "Min" == op_name:
            inputs = []
            for _in in node_inputs:
                data = tf_tensor[_in] if _in in tf_tensor else operators.TorchWeights2TF(onnx_weights[_in])
                inputs.append(data)
            tf_tensor[node_outputs[0]] = tf.keras.layers.Minimum()(inputs)
        elif "Max" == op_name:
            inputs = []
            for _in in node_inputs:
                data = tf_tensor[_in] if _in in tf_tensor else operators.TorchWeights2TF(onnx_weights[_in])
                inputs.append(data)
            tf_tensor[node_outputs[0]] = tf.keras.layers.Maximum()(inputs)
        elif "ReduceMean" == op_name:
            axes = op_attr['axes']
            if 'keepdims' in op_attr:
                keepdims = op_attr['keepdims'] == 1
            else:
                keepdims = False
            axes = [operators.Torch2TFAxis(a) for a in op_attr['axes']]
            tf_tensor[node_outputs[0]] = tf.reduce_mean(tf_tensor[node_inputs[0]], axis=tuple(axes), keepdims=keepdims)
        elif "ReduceSum" == op_name:
            axes = op_attr['axes']
            if 'keepdims' in op_attr:
                keepdims = op_attr['keepdims'] == 1
            else:
                keepdims = False
            axes = [operators.Torch2TFAxis(a) for a in op_attr['axes']]
            tf_tensor[node_outputs[0]] = tf.reduce_sum(tf_tensor[node_inputs[0]], axis=tuple(axes), keepdims=keepdims)
        elif "ReduceMax" == op_name:
            axes = op_attr['axes']
            if 'keepdims' in op_attr:
                keepdims = op_attr['keepdims'] == 1
            else:
                keepdims = False
            axes = [operators.Torch2TFAxis(a) for a in op_attr['axes']]
            tf_tensor[node_outputs[0]] = tf.reduce_max(tf_tensor[node_inputs[0]], axis=tuple(axes), keepdims=keepdims)
        elif "Reciprocal" == op_name:
            tf_tensor[node_outputs[0]] = 1/tf_tensor[node_inputs[0]]
        elif "Sqrt" == op_name:
            tf_tensor[node_outputs[0]] = tf.sqrt(tf_tensor[node_inputs[0]])
        elif "Reshape" == op_name:
            out_shape = onnx_weights[node_inputs[1]]
            shape_len = len(tf_tensor[node_inputs[0]].shape)
            trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            trans_out = [0] + [n for n in range(2, len(out_shape))] + [1]
            if len(out_shape) == 1:
                trans_in = None
                trans_out = None
            tf_tensor[node_outputs[0]] = operators.TFReshape(out_shape)(tf_tensor[node_inputs[0]], trans_in, trans_out)
        elif "Split" == op_name:
            axis = op_attr.get('axis', -1)
            if axis < 0:
                axis = len(tf_tensor[node_inputs[0]].shape) + axis
            axis = operators.Torch2TFAxis(axis)
            if op_attr.get('split'):
                # split 在属性里面的时候
                for index in range(len(op_attr['split'])):
                    tf_tensor[node_outputs[index]] = operators.TFSplit(op_attr, index=index, axis=axis)(tf_tensor[node_inputs[0]])
            else:
                # split 在输入里面的时候
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
                num_or_size_splits = len(node_outputs)
                if (len(node_inputs)>1 and node_inputs[1] in onnx_weights.keys()):
                    num_or_size_splits = onnx_weights[node_inputs[1]].tolist()
                temp = tf.split(tf_tensor[node_inputs[0]], num_or_size_splits=num_or_size_splits, axis=axis)
                for index in range(len(node_outputs)):
                    tf_tensor[node_outputs[index]] = temp[index]
        elif "Transpose" == op_name:
            if len(op_attr['perm']) <= 4:
                shape = list(op_attr['perm'])
                shape_len = len(tf_tensor[node_inputs[0]].shape)
                trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
                trans_out = [0] + [n for n in range(2, len(shape))] + [1]
                temp = operators.TFTranspose(trans_in)(tf_tensor[node_inputs[0]])
                temp = operators.TFTranspose(shape)(temp)
                tf_tensor[node_outputs[0]] = operators.TFTranspose(trans_out)(temp)
            else:
                shape = []
                for axis in op_attr['perm']:
                    new_axis = operators.Torch2TFAxis(axis)
                    if new_axis == -1:
                        new_axis = max(op_attr['perm'])
                    shape.append(new_axis)
                shape = operators.TorchShape2TF(shape)
                tf_tensor[node_outputs[0]] = operators.TFTranspose(shape)(tf_tensor[node_inputs[0]])
        elif "Pow" == op_name:
            tf_tensor[node_outputs[0]] = tf.pow(tf_tensor[node_inputs[0]], onnx_weights[node_inputs[1]])
        elif "Exp" == op_name:
            tf_tensor[node_outputs[0]] = tf.exp(tf_tensor[node_inputs[0]])
        elif "Softplus" == op_name:
            tf_tensor[node_outputs[0]] = tf.math.log(tf.exp(tf_tensor[node_inputs[0]]) + 1)
        elif "Clip" == op_name:
            if "min" in op_attr:
                clip_min = op_attr['min']
            else:
                clip_min = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else onnx_weights[node_inputs[1]]
            if "max" in op_attr:
                clip_max = op_attr['max']
            else:
                clip_max = tf_tensor[node_inputs[2]] if node_inputs[2] in tf_tensor else onnx_weights[node_inputs[2]]
            tf_tensor[node_outputs[0]] = operators.TFClip(clip_min, clip_max)(tf_tensor[node_inputs[0]])
        elif "Expand" == op_name:
            shape = None
            if node_inputs[1] in tf_tensor:
                shape = tf_tensor[node_inputs[1]]
            elif node_inputs[1] in onnx_weights:
                shape = operators.TorchShape2TF(onnx_weights[node_inputs[1]])
            else:
                shape = operators.TorchShape2TF(op_attr[node_inputs[1]])
            inputs = tf_tensor[node_inputs[0]]
            for i in range(len(shape)):
                if int(shape[i]//inputs.shape[i]) > 1:
                    inputs = tf.repeat(inputs, repeats=int(shape[i]//inputs.shape[i]), axis=i)
            tf_tensor[node_outputs[0]] = inputs
        elif "GatherElements" == op_name:
            indices = tf_tensor[node_inputs[1]] if node_inputs[1] in tf_tensor else operators.TorchWeights2TF(onnx_weights[node_inputs[1]])
            axis = op_attr.get("axis", 0)
            tf_tensor[node_outputs[0]] = tf.experimental.numpy.take_along_axis(data, indices=indices, axis=axis)
        elif "GatherND" == op_name:
            indices = tf_tensor[node_inputs[1]]
            shape_len = len(tf_tensor[node_inputs[0]].shape)
            trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            
            temp = operators.TFTranspose(trans_in)(tf_tensor[node_inputs[0]])
            temp = tf.gather_nd(temp, indices)
            if len(temp.shape) > 2:
                trans_out = [0] + [n for n in range(2, len(temp.shape))] + [1]
                tf_tensor[node_outputs[0]] = operators.TFTranspose(trans_out)(temp)
            else:
                tf_tensor[node_outputs[0]] = temp
        elif "HardSigmoid" == op_name:
            alpha = op_attr.get("alpha", 0.2)
            beta = op_attr.get("beta", 0.5)
            tf_tensor[node_outputs[0]] = tf.clip_by_value(tf_tensor[node_inputs[0]]*alpha + beta, 0, 1) 
        elif "HardSwish" == op_name:
            alpha = 1/6
            beta = 0.5
            tf_tensor[node_outputs[0]] = tf_tensor[node_inputs[0]]*tf.clip_by_value(tf_tensor[node_inputs[0]]*alpha + beta, 0, 1)
        elif "Identity" == op_name:
            tf_tensor[node_outputs[0]] = tf_tensor[node_inputs[0]]
        elif "Dropout" == op_name:
            tf_tensor[node_outputs[0]] = tf_tensor[node_inputs[0]]
        else:
            raise Exception("operator {} is not implemented yet.".format(op_name))

    keras_model = keras.Model(inputs=[tf_tensor[x.name] for x in model_graph.input], outputs=[tf_tensor[x.name] for x in model_graph.output])
    keras_model.trainable = False
    keras_model.summary()
    del model_graph
    del onnx_weights

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
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
    if tflite_model_name is None:
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


if __name__ == "__main__":
    # ONNX2TFLiteConvertor("/home/test/yolov8n.onnx",
    #                      "/home/test/",
    #                      None,
    #                      need_simplify=False,
    #                      is_quant=False,
    #                      is_int8=False,
    #                      image_root="/data/zhangxian/data/coco/images/val2017/",
    #                     #  int8quant_mean=None,
    #                     #  int8quant_std=None)
    #                     #  int8quant_mean=[123.675, 116.28, 103.53],
    #                     #  int8quant_std=[58.395, 57.12, 57.375])
    #                      int8quant_mean=[127.5, 127.5, 127.5],
    #                      int8quant_std=[127.5, 127.5, 127.5])
    # onnx 版本不能超过12
    ONNX2TFLiteConvertor(r'D:\algorithm\ultralytics\tflite\yolov8n.onnx',
                         r'D:\algorithm\ultralytics\tflite',
                         None,
                         need_simplify=False,
                         is_quant=False,
                         is_int8=False,
                         image_root=r"D:\algorithm\datasets\coco128\images\train2017",
                        #  int8quant_mean=None,
                        #  int8quant_std=None)
                        #  int8quant_mean=[123.675, 116.28, 103.53],
                        #  int8quant_std=[58.395, 57.12, 57.375])
                         int8quant_mean=[127.5, 127.5, 127.5],
                         int8quant_std=[127.5, 127.5, 127.5])

    