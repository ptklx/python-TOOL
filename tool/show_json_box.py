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

def showimgfile(testfile,input_json_path):
    with open(input_json_path, 'r', encoding='utf8') as fp:
        input_json_data = json.load(fp)
        images_list = input_json_data["images"]
        labels_list = input_json_data["annotations"]
        # for index, image_name in enumerate(images_list):
        for index , label in enumerate(labels_list):
            bbox = label["bbox"]
            id = label['id']
            category_id = label["category_id"]
            image_id = label["image_id"]
            file_name=''
            for image in images_list:
                if image_id == image['id']:  
                    file_name = image['file_name']
                    break
            if file_name=='':
                continue
            img_path = os.path.join(testfile,file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
            h,w,_ = img.shape
            x,y,width,height=bbox
            # cenx =stx*w
            # ceny =sty*h
            # boxw = ex*w
            # boxh = ey*h
            # startx = int(cenx-boxw/2)
            # starty = int(ceny-boxh/2)
            startx = int(x/2)
            starty = int(y/2)
            endx = int((x+width)/2)
            endy = int((y+height)/2)
            cv2.rectangle(img,(startx,starty),(endx,endy),(0,255,0),2)
            cv2.imshow("test",img)
            cv2.waitKey(0)


        

if __name__ == "__main__":
    # testfile =r"E:\data1\train\from\retailproduct\test2019"
    # input_json_path =r"E:\data1\train\from\retailproduct\instances_test2019.json"   #

    testfile =r"E:\data1\train\from\retailproduct\val2019"
    input_json_path =r"E:\data1\train\from\retailproduct\instances_val2019.json"   #

    # testfile =r"E:\data1\Retail Product Checkout Dataset\retail_product_checkout_zips\train2019"
    # input_json_path =r"E:\data1\Retail Product Checkout Dataset\retail_product_checkout_zips\instances_train2019.json"   #
    showimgfile(testfile,input_json_path)
    