#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import numpy as np
import cv2
from shutil import copyfile
# import shutil
import math
import re
import json

def showimgfile(testfile,input_json_path,text_label):
    with open(input_json_path, 'r', encoding='utf8') as fp:
        input_json_data = json.load(fp)
        images_list = input_json_data["images"]
        labels_list = input_json_data["annotations"]
        outpic_labels_path = os.path.join(os.path.split(testfile)[0],text_label)

        if not os.path.exists(outpic_labels_path):
            os.makedirs(outpic_labels_path, exist_ok=True)
        for index, images_info in enumerate(images_list):
            boxs_list =[]
            width = images_info["width"]
            height = images_info["height"]
            for label in labels_list:
                if label["image_id"] == images_info['id']:  
                    x,y,w,h=bbox = label["bbox"]
                    bbox_t =[(x+w/2)/width,(y+h/2)/height,w/width,h/height]
                    boxs_list.append(bbox_t)
            stri_t = images_info['file_name']
            savelabelsname = os.path.join(outpic_labels_path, stri_t.replace(".jpg",".txt"))
            with open(savelabelsname,'w') as f:
                    for label in boxs_list:
                        line = ' '.join(str(round(coord,6))for coord in label)
                        f.write('0 '+ line +'\n')
    

if __name__ == "__main__":
    testfile =r"E:\data1\train\from\retailproduct\test2019"
    input_json_path =r"E:\data1\train\from\retailproduct\instances_test2019.json"   #
    text_label = "test_labels"

    # testfile =r"E:\data1\train\from\retailproduct\val2019"
    # input_json_path =r"E:\data1\train\from\retailproduct\instances_val2019.json"   #
    # text_label = "val_labels"
    # testfile =r"E:\data1\Retail Product Checkout Dataset\retail_product_checkout_zips\train2019"
    # input_json_path =r"E:\data1\Retail Product Checkout Dataset\retail_product_checkout_zips\instances_train2019.json"   #
    showimgfile(testfile,input_json_path,text_label)
    