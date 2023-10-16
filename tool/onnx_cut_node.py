import onnx

def cut_output():
    input_path =r'D:\algorithm\ultralytics\tflite\yolov8n_12.onnx'
    output_path = r'D:\algorithm\ultralytics\tflite\yolov8n_12_cut.onnx'
    input_names = ['image']

    output_names = ['/model.22/Concat_8_output_0']


    onnx.utils.extract_model(input_path, output_path, input_names, output_names)   # 通过输出删除结尾部分
    print("ok!")


def remove_preprocessing_node(onnx_path):
    onnx_model = onnx.load(onnx_path) #加载onnx模型
    graph = onnx_model.graph
    old_nodes = graph.node
    new_nodes = old_nodes[2:] #去掉data,sub,mul前三个节点
    del onnx_model.graph.node[:] # 删除当前onnx模型的所有node
    onnx_model.graph.node.extend(new_nodes) # extend新的节点
    conv0_node = onnx_model.graph.node[0]
    conv0_node.input[0] = 'data' #给第一层的卷积节点设置输入的data节点
    # graph = onnx_model.graph
    # print(graph.node)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "./janus_mask_occ_nopre.onnx")


def sub_node():
    # 导入resnet50.onnx模型
    onnx_path =r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify_test.onnx"
    out_onnx_path =r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify_test1.onnx"
    resnet50_onnx = onnx.load(onnx_path)
    # 获得onnx图
    graph = resnet50_onnx.graph
    # 获得onnx节点
    node = graph.node

    ### 准备工作已就绪，开干
    # 增、删、改、查一起操作
    # 比如咱们要对 `算子类型为Add&输出为225的节点` 进行操作
    need_del_list =[]
    input_name_list=[]
    output_name_list=[]

    for i in range(len(node)):
        n = i - len(need_del_list)  # 避免溢出
        if node[n].op_type == 'Transpose':   #最好也限制下
            node_rise = node[n]
            input_name_list.append(node_rise.input[0])
            output_name_list.append(node_rise.output[0])

            # if node_rise.output[0] == '225':
                # print(i)  # 169 => 查到这个算子的ID为169
            need_del_list.append(n)
            node.remove(node[n])   #删除一个后node对应位置会变化

    for i in range(len(node)):  #由于删除后，输入输出节点也需要更改下
        # n = i - len(need_del_list)  # 避免溢出
        if node[i].op_type == 'Sigmoid':
            node_rise = node[i]
            if node_rise.input[0] in output_name_list:
                node_rise.input[0]= input_name_list[output_name_list.index(node_rise.input[0])]


    # old_node = node[169]  # 定位到刚才检索到的算子
    # 删除旧节点
    # node.remove(old_node)  
 
         

    # 新增一个 `Constant` 算子
    # new_node = onnx.helper.make_node(
    #     "Constant",
    #     inputs=[],
    #     outputs=['225'],
    #     value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [4], [1, 1, 1.2, 1.2])
    # )  




    # 插入新节点
    # node.insert(169, new_node)  

    # 是不是还少一个修改节点，比方看下面
    # node[169].type = 'Conv'   # 将刚才的算子类型改为2D卷积
    # 改名称啥的类似

    ### 保存新模型
    # 校验
    onnx.checker.check_model(resnet50_onnx)
    # 保存
    onnx.save(resnet50_onnx,out_onnx_path)



if __name__ == "__main__":
    cut_output()  # 裁剪结尾
    # sub_node()

