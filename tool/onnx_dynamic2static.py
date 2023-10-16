import onnx
from onnx.tools import update_model_dims

# path =r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference.onnx"
# out_path=r"D:\algorithm\retail_project\cart_shopping_rec_one2one\paddle_mainbody\temp2\inference_modify.onnx"

path =r"D:\algorithm\retail_project\cart_shopping_rec_one2one\model\inference.onnx"
out_path=r"D:\algorithm\retail_project\cart_shopping_rec_one2one\model\inference_modify.onnx"
model = onnx.load(path)
inputs = model.graph.input
outputs = model.graph.output
inputs[0].type.tensor_type.shape.dim[0].dim_value=1
outputs[0].type.tensor_type.shape.dim[0].dim_value=1
# inputs[1].type.tensor_type.shape.dim[0].dim_value=1
model= onnx.shape_inference.infer_shapes(model)
# updated_model = update_model_dims.update_inputs_outputs_dims(model, {"input1":[1,3,112,112]}, {"input2":[1, 3, 112, 112]})
onnx.save(model, out_path)