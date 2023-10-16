'''
    ONNX->Keras, Keras基本对应算子
    author: 肖禾
    time: 2022/01/07
'''
import tensorflow as tf
from tensorflow import keras

class TFBatchNormalization(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, weight, bias, running_mean, running_var, epsilon=1e-5, momentum=0.9):
        super().__init__()
        epsilon = 1e-5 if epsilon is None else epsilon
        momentum = 0.9 if epsilon is None else momentum
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(bias),
            gamma_initializer=keras.initializers.Constant(weight),
            moving_mean_initializer=keras.initializers.Constant(running_mean),
            moving_variance_initializer=keras.initializers.Constant(running_var),
            epsilon=epsilon,
            momentum=momentum)

    def call(self, inputs):
        return self.bn(inputs)

class TFInstanceNormalization():
    # TensorFlow InstanceNormalization wrapper
    def __init__(self, scale, bias, epsilon=1e-5):
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.epsilon = 1e-5 if epsilon is None else epsilon

    def __call__(self, inputs):
        lens = len(inputs.shape)
        axis = tuple(range(1, lens-1))
        mean = tf.reduce_mean(inputs, axis=axis, keepdims=True)
        var = tf.math.reduce_variance(inputs, axis= axis, keepdims=True)
        inputs = self.scale*(inputs - mean)/tf.sqrt(var + self.epsilon) + self.bias
        return inputs

class TFPad(keras.layers.Layer):
    def __init__(self, pad, model="constant"):
        super().__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        self.model = model

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode=self.model)

class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, dilations=1, pads=None, g=1, w=None, b=None):
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(s, int):
            s = (s, s)
        if dilations[0] != 1 and s[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
        conv_model_str = 'SAME' if s == 1 and k == 1 else "VALID"
        self.conv = keras.layers.Conv2D(
            # c2, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False if b is None else True,
            c2, k, s, conv_model_str, use_bias=False if b is None else True,
            kernel_initializer=keras.initializers.Constant(w),
            bias_initializer='zeros' if b is None else keras.initializers.Constant(b),
            # groups=g,
            dilation_rate=dilations)
        
        if pads is None or max(pads) == 0:
            self.pad = None
        else:
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.ZeroPadding2D(padding=padding)

    def call(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.conv(inputs)

class TFGroupConv(keras.layers.Layer):
    # Group Convolution
    def __init__(self, cin, cout, k=1, s=1, dilations=1, pads=None, groups=1, w=None, b=None):
        super().__init__()
        filters = w.shape[-2]
        assert groups*filters == cout, "Input channels and filters must both be divisible by groups."
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(s, int):
            s = (s, s)
        if dilations[0] != 1 and s[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
        self.cin = cin
        self.groups = groups
        cout = int(cout//groups)
        if pads is None:
            self.pad = None
        else:
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.ZeroPadding2D(padding=padding)
        
        self.convs = []
        for i in range(groups):
            self.convs.append(keras.layers.Conv2D(
                                cout, k, s, 'VALID', use_bias=False if b is None else True,
                                dilation_rate=dilations,
                                kernel_initializer=keras.initializers.Constant(w[:, :, :, i*cout:(i+1)*cout]),
                                bias_initializer='zeros' if b is None else keras.initializers.Constant(b[i*cout:(i+1)*cout])))

    def call(self, inputs):
        if self.pad is not None:
            inputs = self.pad(inputs)
        outs = []
        in_s = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
        for i in range(self.groups):
            outs.append(self.convs[i](in_s[i]))
        outs = tf.concat(outs, axis=-1)
        return outs

class TFDepthwiseConv2D(keras.layers.Layer):
    def __init__(self, k=1, s=1, dilations=1, pads=None, w=None, b=None) -> None:
        super().__init__()

        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(s, int):
            s = (s, s)
        self.conv = keras.layers.DepthwiseConv2D(
            k, s, "VALID", use_bias=False if b is None else True,
            weights=[w] if b is None else [w, b],
            dilation_rate=dilations,
            activation=None,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )

        if pads is None or max(pads) == 0:
            self.pad = None
        else:
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.ZeroPadding2D(padding=padding)
            
    def call(self, inputs):
        if self.pad is not None:
            inputs = self.pad(inputs)
        return self.conv(inputs)

class TFConvTranspose(keras.layers.Layer):
    def __init__(self, c1, c2, k=1, s=1, dilations=1, pads=None, g=1, w=None, b=None):
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        if isinstance(s, int):
            s = (s, s)
        if dilations[0] != 1 and s[0] != 1:
            raise Exception("Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.")
        conv_model_str = 'SAME' if s == 1 and k == 1 else "VALID"
        self.conv = keras.layers.Conv2DTranspose(
                                        c2, k, s, conv_model_str, use_bias=False if b is None else True,
                                        kernel_initializer=keras.initializers.Constant(w),
                                        bias_initializer='zeros' if b is None else keras.initializers.Constant(b),
                                        # groups=g,
                                        dilation_rate=dilations)
        
        if pads is None or max(pads) == 0:
            self.pad = None
        else:
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))
            self.pad = keras.layers.ZeroPadding2D(padding=padding)

    def call(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)
        return self.conv(inputs)

class TFSlice(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs, starts, ends, axes, steps):
        indices = tf.keras.backend.arange(starts, ends, step=steps)

        return tf.gather(inputs, indices, axis=axes)

class TFConcat(keras.layers.Layer):
    def __init__(self, dimension=1, w=None):
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3
    def call(self, inputs):
        return tf.concat(inputs, self.d)

class TFScatterND(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs, indices, updates):
        return tf.tensor_scatter_nd_update(inputs, indices, updates)

class TFShape(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs):
        return inputs.shape

class TFAveragePool(keras.layers.Layer):
    def __init__(self, kernel_shape=1, pads=None, strides=1) -> None:
        super().__init__()
        if pads is None:
            self.pad = tf.identity
        else:
            self.pad = TFPad(pads[0])
        self.avg_pool = keras.layers.AveragePooling2D(pool_size=kernel_shape, strides=strides, padding='VALID')
    
    def call(self, inputs):
        return self.avg_pool(self.pad(inputs))

class TFMaxPool(keras.layers.Layer):
    def __init__(self, kernel_shape=1, pads=None, strides=1) -> None:
        super().__init__()
        if pads is None:
            self.pad = tf.identity
        else:
            self.pad = TFPad(pads[0])
        self.max_pool = keras.layers.MaxPool2D(pool_size=kernel_shape, strides=strides, padding='VALID')
    
    def call(self, inputs):
        return self.max_pool(self.pad(inputs))

class TFGather(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs, indices, axis):
        return tf.gather(inputs, indices, axis=axis)

class TFCast(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs, target_type):
        return inputs

class TFFloor(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs):
        return tf.floor(inputs)

class TFUnsequeeze(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
    def call(self, inputs, axis):
        return tf.expand_dims(inputs, axis)

class TFGlobalAveragePool():
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, inputs):
        axes = [i for i in range(1, len(inputs.shape)-1)]
        return tf.reduce_mean(inputs, axis=axes, keepdims=True)

class TFGlobalMaxPool():
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, inputs):
        axes = [i for i in range(1, len(inputs.shape)-1)]
        return tf.reduce_max(inputs, axis=axes, keepdims=True)

class TFFlatten(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def call(self, inputs):
        tensor_size, tensor_shape = 1, inputs.get_shape().as_list()
        for n in tensor_shape[1:]:
            tensor_size = tensor_size * max(n, 1)

        if tensor_size == max(tensor_shape[1:]):
            return tf.reshape(inputs, shape=(tensor_shape[0], -1))
        else:
            perm_list = [0, len(tensor_shape)-1]
            for i in range(len(tensor_shape)-2):
                perm_list.append(i+1)
            inputs = tf.transpose(inputs, perm=perm_list)
            return tf.reshape(inputs, shape=(tensor_shape[0], -1))

class TFReshape(keras.layers.Layer):
    def __init__(self,outshape):
        super().__init__()
        self.outShape = tuple(outshape)

    def __call__(self, inputs, trans_in, trans_out):
        if trans_in:
            inputs = tf.transpose(inputs, perm=trans_in)
        inputs = tf.reshape(inputs, shape=self.outShape)
        if trans_out:
            inputs = tf.transpose(inputs, perm=trans_out)
        return inputs

class TFTranspose(keras.layers.Layer):
    def __init__(self, perm_list)->None:
        super().__init__()
        self.perm_list = perm_list

    def __call__(self, inputs):
        return tf.transpose(inputs, perm=self.perm_list)

class TFClip(keras.layers.Layer):
    def __init__(self, min, max)->None:
        super().__init__()
        self.min = min
        self.max = max

    def __call__(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)

class TFGemm(keras.layers.Layer):
    '''
        全连接函数, torch.linear, tf.layers.dense, keras.layers.Dense
    '''
    def __init__(self, weights, bias=None) -> None:
        super().__init__()
        weights = weights.T
        bias = bias[None, ...]
        if bias is None:
            self.dense = keras.layers.Dense(weights.shape[1],
                                            input_shape=(weights.shape[0],),
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=keras.initializers.Constant(weights))
        else:
            self.dense = keras.layers.Dense(weights.shape[1],
                                            input_shape=(weights.shape[0],),
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=keras.initializers.Constant(weights),
                                            bias_initializer=keras.initializers.Constant(bias))
    def call(self, inputs):
        return self.dense(inputs)

class TFSplit():
    def __init__(self, node_attribute, index=0, axis=-1):
        super().__init__()
        start = 0
        for i in range(index):
            start += int(node_attribute['split'][i])
        end = start + node_attribute['split'][index]
        self.indices = tf.keras.backend.arange(start, end, 1)
        self.axis = axis

    def __call__(self, inputs):
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

def Torch2TFAxis(axis):
    if axis == 0:
        axis = 0
    elif axis == 1:
        axis = -1
    elif axis < 0:
        axis = axis
    else:
        axis -= 1
    return axis

def TorchShape2TF(shape:list or tuple):
    if len(shape) <= 2:
        return tuple(shape)
    new_shape = [shape[0], *shape[2:], shape[1]]
    return tuple(new_shape)

def TorchWeights2TF(weights):
    if(len(weights.shape) > 2):
        shape = [i for i in range(len(weights.shape))]
        shape = TorchShape2TF(shape)
        weights = weights.transpose(*shape)
    return weights

def convertType(x:tf.Variable):
    if x.dtype == tf.float32:
        return x
    return tf.cast(x, dtype=tf.float32)

if __name__ == '__main__':
    import numpy as np
    x = np.ones((2, 3, 24, 32))
    op = TFReshape((2, -1))
    out = op(x)
    print(out.shape)