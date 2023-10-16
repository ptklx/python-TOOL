#!/usr/bin/python
# -*- coding: UTF-8 -*-
# import sys
import os
# import numpy as np
import cv2
# from shutil import copyfile
# import shutil
# import math
# import re
import json

def showimgfile(testfile,input_json_path,text_label):
    if not os.path.isdir(text_label):
            os.makedirs(text_label)
    # if not os.path.exists(text_label):
        # os.makedirs(text_label, exist_ok=True)
    files = os.listdir(testfile)
    for fi in files:
        fi_d = os.path.join(testfile,fi)
        if os.path.isfile(fi_d):
            # savel = fi.split('_')[1]
            name = os.path.splitext(fi)[0]
            suffix = os.path.splitext(fi)[1]
            json_path = os.path.join(input_json_path,name+".json")
            out_text_path = os.path.join(text_label,name+".txt")
            if suffix ==".jpg" :    #or suffix==".png":
                im0s = cv2.imread(fi_d)
                heighty,widthx,_ = im0s.shape
                with open(json_path, 'r', encoding='utf8') as fp:
                    input_json_data = json.load(fp)
                    box_info = input_json_data["shapes"]
                    boxs_list =[]

                    for box in box_info:
                        label_flag = int(box["label"])-1
                        # label_flag=0
                        xy_points = box["points"]
                        x1,y1 = xy_points[0]
                        x2,y2 = xy_points[1]
                        bbox_t =[label_flag,((x1+x2)/2)/widthx,((y1+y2)/2)/heighty,abs(x2-x1)/widthx,abs(y2-y1)/heighty]
                        boxs_list.append(bbox_t)

                        #test
                        # cv2.rectangle(im0s, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                    # frame = cv2.resize(im0s, (0, 0), fx=0.5, fy=0.5)
                    # cv2.imshow('image', frame)  
                    # cv2.waitKey(0)
                    #end test
                    # stri_t = images_info['file_name']
                    # savelabelsname = os.path.join(outpic_labels_path, stri_t.replace(".jpg",".txt"))
                    savelabelsname = out_text_path
                    with open(savelabelsname,'w') as f:
                            for label in boxs_list:
                                line = ' '.join(str(round(coord,6))for coord in label)
                                # f.write(label_flag+' '+ line +'\n')
                                f.write(line +'\n')
    

if __name__ == "__main__":
    testfile =r"E:\data1\train\from\yolov5_cart\all_img"
    input_json_path =r"E:\data1\train\from\yolov5_cart\all_json"   #
    text_label = r"E:\data1\train\from\yolov5_cart\test_labels"


    showimgfile(testfile,input_json_path,text_label)
    