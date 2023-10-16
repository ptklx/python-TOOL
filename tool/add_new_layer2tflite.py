import tensorflow as tf
import numpy as np
# 加载原始的TFLite模型
tflite_model_path = r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify_test1.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# signatures_details = interpreter.get_signature_list()
tensor_details = interpreter.get_tensor_details()
need_add_layer=[2,3,4,6]
# 创建新的全连接层
# dense_layer = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)

# dense_layer =tf.keras.layers.Activation(tf.nn.sigmoid,dtype = "float")
# 获得新层的输入和输出张量
# new_input_tensor = dense_layer(input_details[0]['shape'])
# new_output_tensor = dense_layer.output
# 将新层添加到模型中
# interpreter.set_tensor(input_details[0]['index'], new_input_tensor)


# for i in need_add_layer:
#     output_index = output_details[i]['index']  
#     output_data = interpreter.get_tensor(output_index)
 
#     sigmoid_output = 1 / (1 + np.exp(-output_data))  # 添加sigmoid 还有问题
#     sigmoid_output= sigmoid_output if isinstance(sigmoid_output,np.ndarray) else sigmoid_output.numpy()
#     # new_input_tensor = dense_layer(output_details[i]['shape'])
#     # new_output_tensor = dense_layer.output
#     # interpreter.allocate_tensors()
#     interpreter.set_tensor(output_details[i]['index'], sigmoid_output)

# output_index = output_details[2]['index']  
# weight_data = interpreter.tensor(output_index)() 
# sigmoid_data = 1 / (1 + np.exp(-weight_data))
# sigmoid_data= sigmoid_data if isinstance(sigmoid_data,np.ndarray) else sigmoid_data.numpy()
# # interpreter.tensor(output_index)()[...] = sigmoid_data
# interpreter.set_tensor(output_index, sigmoid_data)
# 运行模型
interpreter.invoke()



# 获取新层的输出
# new_output = interpreter.get_tensor(output_details[0]['index'])

# 保存修改后的模型
modified_model_path = r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify_test1_modified.tflite"
# with open(modified_model_path, 'wb') as f:
    # f.write(interpreter.tensor(interpreter.get_tensor_details()))

# 量化模型  
quantizer = tf.lite.experimental.Quantization.create_default_quantizer()  
quantized_model = quantizer.quantize(interpreter)  
  
# 保存量化后的模型  
with open(modified_model_path, 'wb') as f:  
    f.write(quantized_model)

  

print("ok!")
