import os
import sys
import subprocess
# __dir__ = os.path.dirname(os.path.abspath(__file__))
__dir__ = r"D:\file\ocr_detect\PaddleOCR\tools"
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import numpy as np
import json
import time
from PIL import Image
sys.path.append(r"D:\file\ocr_detect\PaddleOCR")
import tools.infer.utility as utility
import tools.infer.predict_det as predict_det
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
logger = get_logger()

labelme_version = '5.3.1'


def rmqrm(filepath):
    p = open(filepath, 'r+')
    lines = p.readlines()
    d = ""
    for line in lines:
        c = line.replace('"group_id": "null",', '"group_id": null,')
        d += c

    p.seek(0)
    p.truncate()
    p.write(d)
    p.close()

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_detector = predict_det.TextDetector(args)
    if not os.path.exists(args.out_label_path):
        os.makedirs(args.out_label_path)

    if not os.path.exists(args.save_train_img):
        os.makedirs(args.save_train_img)
    
    for idx, image_file in enumerate(image_file_list):
        img = cv2.imread(image_file)
        img=cv2.resize(img, None, fx=0.5, fy=0.5)   ##
        # img=cv2.resize(img,(640,480))   ##  w,h          #######################################################static
        starttime = time.time()
        dt_boxes,loc_t= text_detector(img)
        elapse = time.time() - starttime
        # print(dt_boxes)

        shapes = []
        for box in dt_boxes:
            vertis = box.copy()
            # vertis=vertis.astype(np.int32)
            # cv2.polylines(img, [vertis], isClosed=True, color=(255, 255, 0), thickness=3)
            shape = dict()
            shape['label'] = '1'
            shape['points'] = box.tolist()
            shape['group_id'] = 'null'
            shape['description'] = ''
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shapes.append(shape)

        # cv2.imshow("test",img)
        # cv2.waitKey(0)
        
        labelShape= img.shape 
        
        obj = dict()
        obj['version'] = labelme_version
        obj['flags'] = {}
            # print(len(shapes))
        obj['shapes'] = shapes
        # print(shapes)
        (_,imgname) = os.path.split(image_file)
        obj['imagePath'] = "../train_img/"+imgname   #这个文件名很重要，是通过这个访问
        # print(obj['imagePath'])
        # obj['imageData'] = str(imgEncode(oriImgPath))
        obj['imageData'] = None #str(imgEncode(oriImgPath))

        obj['imageHeight'] = labelShape[0]
        obj['imageWidth'] = labelShape[1]

        j = json.dumps(obj, sort_keys=True, indent=4)
        save_path =os.path.join(args.out_label_path,os.path.split(image_file)[1][:-4]+".json")
        # print(saveJsonPath)
        with open(save_path, 'w') as f:
            f.write(j)
        rmqrm(save_path)
        save_img_p =os.path.join(args.save_train_img,imgname)
        cv2.imwrite(save_img_p,img)

 


if __name__ == "__main__":
    args = utility.parse_args()
    #######add param
    args.use_gpu = False
    args.use_onnx =True
    # args.use_angle_cls=True  # 旋转下反而有个识别差了  
    args.det_model_dir = r"D:\file\ocr_detect\PaddleOCR\paddle_weight\ch-pp-ocrv4-server\ch_PP-OCRv4_det_server_infer\det_server.onnx"
    args.image_dir=r"E:\data2\paddle_ocr_det_img\our_collect\bianz"
    # args.image_dir=r"D:\file\ocr_detect\snapshot\min.jpg"
    
    args.out_label_path=r"E:\data2\paddle_ocr_det_img\our_collect\jsonlable"
    args.save_train_img =r"E:\data2\paddle_ocr_det_img\our_collect\train_img"
    main(args)
