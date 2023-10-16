import os
# import cv2.dnn
import cv2
import numpy as np

import tensorflow as tf
# import tflite_runtime.interpreter as tflite	# 改动一
import sys


# from aidlux_utils import draw_detect_res,postprocess_old,convert_shape,NMS



sys.path.append(r"D:\algorithm\ultralytics")

# from ultralytics.utils import ROOT, yaml_load
# from ultralytics.utils.checks import check_requirements, check_yaml


#yolov5  conda activate base
###conda activate ai_create     



classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", \
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",\
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", \
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", \
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", \
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


class picodet_predict(object):
    def __init__(self,tflite_model,threshold = 0.45) -> None:
   
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model)
        # self.interpreter = tflite.Interpreter(tflite_model)   # 改动二
        self.interpreter.allocate_tensors()

        # 获取输入和输出张量的索引
        self.input_details = self.interpreter.get_input_details()
        _,self.input_height,self.input_width,self.channel=self.input_details[0]["shape"]   #
        # self.input_width = 640
        # self.input_height = 640

        self.output_details = self.interpreter.get_output_details()

        # self.model_size = (640, 640)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.scale = 1/255.

        self.confidence_thres = 0.5 #threshold
        self.iou_thres = 0.5
        # Generate a color palette for the classes
        # Load the class names from the COCO dataset
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def predict(self,img,rgb=True):

        # Convert the image color space from BGR to RGB
        img_ori = img.copy()

        if rgb:
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)  #[:, :, ::-1] 同理

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        # image_data = np.array(img) / 255.0

        # image_data = (img.astype('float32') * self.scale - self.mean) / self.std
        image_data = img.astype('float32')  / 255.0
        # Transpose the image to have the channel dimension as the first dimension
        # image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
     

        # Expand the dimensions of the image data to match the expected input shape
        img_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Run inference using the preprocessed image data
        # outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        self.interpreter.set_tensor(self.input_details[0]['index'], img_data)

        # 进行推理
        self.interpreter.invoke()

        # 获取输出结果
        outputs =[]
        # for i  in range(8):
        #     start = 0 
        #     for idx , output in enumerate(self.output_details):
        #         if str(i) in output["name"]:
        #             start = idx
        # for output in self.output_details:
            # outputs.append(self.interpreter.get_tensor(output['index']))
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])

        output_img = self.postprocess(img_ori, outputs)

        # boxes,scores,class_ids = postprocess_old(outputs, [1., 1.],conf_thres=0.2,iou_thres=0.3)

        # scores=np.expand_dims(scores,1)
        # class_ids=np.expand_dims(class_ids,1)
        # bboxes=np.concatenate((boxes,scores),1)
        # bboxes =np.concatenate((bboxes,class_ids),1)
        # print("tflite--bboxes--",bboxes)
        
        # cost_time = 0.8 * cost_time + 0.2 * (time.time() - t0)
        # print("Avg_FPS:{:^4.2f},OnTime_FPS:{:^4.2f}, 检测到{}个区域".format(1 / cost_time, 1 / (time.time() - t0), len(bboxes)))
        
        # res_img = aidlux_utils_bak.draw_detect_res(original_image, bboxes)
        # output_img = draw_detect_res(img_ori, bboxes)

        return   output_img
    
    
        
    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        img_height, img_width = input_image.shape[:2]
        # img_width, img_height = input_image.shape[:2]
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        # outputs = np.squeeze(output[0])
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []


        # Calculate the scaling factors for the bounding box coordinates
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            confidence = outputs[i][4]
            if i<20:
                print(confidence)
            if confidence >= 0.25:
                classes_scores = outputs[i][5:]
                # Find the maximum score among the class scores
                # maxScore = np.amax(classes_scores)
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

                # If the maximum score is above the confidence threshold
                if maxScore >= self.confidence_thres:
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)

                    # Extract the bounding box coordinates from the current row
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # box = [outputs[i][0] - (0.5 * outputs[i][2]), outputs[i][1] - (0.5 * outputs[i][3]),
                    # outputs[i][2], outputs[i][3]]

                    # Add the class ID, score, and box coordinates to the respective lists
                    # class_ids.append(class_id)
                    # scores.append(maxScore)
                    # boxes.append([abs(left), abs(top), abs(width), abs(height)])
                    boxes.append([abs(left), abs(top), abs(width), abs(height)])
                    class_ids.append(maxClassIndex)
                    scores.append(maxScore)
                    # boxes1.append(box)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres,0.5)
        
        bboxes=[]
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            # bbox = boxes1[i]
            score = scores[i]
            class_id = class_ids[i]
            # x, y, w, h = float(abs(bbox[0])), float(abs(bbox[1])), float(abs(bbox[2])), float(abs(bbox[3]))
            # box1 =[round(x*x_factor), round(y*y_factor), round(w*y_factor), round(h*y_factor)]
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
            # box.append(score)
            # box.append(class_id)
            # bboxes.append(box)
        # bboxes =np.concatenate((bboxes,class_ids),1)
        # input_image = draw_detect_res(input_image, bboxes)

        # Return the modified input image
        return input_image



def main(tflite_model, image_path):

    onnxruntimecnn = picodet_predict(tflite_model,threshold=0.2)

    file_list = os.listdir(image_path)
    idx = 0
    for fi in file_list:
        path = os.path.join(image_path,fi)
        # path = r"D:\algorithm\yolov5\tflite\assets\bus.jpg"
        print(path)
        frame = cv2.imread(path)
        # frame = cv2.resize(frame, (640, 640))
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        original_image = frame #cv2.resize(original_image, (int(original_image.shape[1]/2),int(original_image.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
        output_img = onnxruntimecnn.predict(original_image)
        # draw_box_opencv(original_image, detections,thre=0.1)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', output_img)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return 

if __name__ =="__main__":

    tflite_path=r"D:\algorithm\yolov5\tflite\yolov5n_fp32.tflite"
    # image_path=r"E:\data1\our_collect\pic0"
    image_path=r"D:\algorithm\yolov5\tflite\assets"

    main(tflite_path,image_path)
