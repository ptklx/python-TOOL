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

def showimgfile(testfile,labelfile):
    files = os.listdir(testfile)
    # files =  sorted(files,key= lambda x: int(x.split('_')[0])) 
    for fi in files:
        fi_d = os.path.join(testfile,fi)
        if os.path.isfile(fi_d):
            # savel = fi.split('_')[1]
            name = os.path.splitext(fi)[0]
            suffix = os.path.splitext(fi)[1]
            if suffix ==".jpg" or suffix==".png":
                img = cv2.imread(fi_d)
                img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
                h,w,_ = img.shape
                label_txt = os.path.join(labelfile,name+".txt")
                boxes = []
                with open(label_txt,'r') as f:
                    txt_lines = f.readlines()
                    for i in txt_lines:
                        box_str = re.split('[\t \n]',i.strip())
                        box = list(map(float, box_str))
                        boxes.append(box)
                # print(boxes)
                for box in boxes:
                    label,stx,sty,ex,ey=box
                    cenx =stx*w
                    ceny =sty*h
                    boxw = ex*w
                    boxh = ey*h
                    startx = int(cenx-boxw/2)
                    starty = int(ceny-boxh/2)

                    cv2.rectangle(img,(startx,starty),(startx+int(boxw),starty+int(boxh)),(0,255,0),2)
                    label_str = str(int(label) )
                cv2.imshow("test",img)
                cv2.waitKey(0)

                # temp= img[:,step_w*i:(step_w*i+step_w)]
                # out_path = os.path.join(outpath,out_name)
                # cv2.imwrite(out_path,temp)

if __name__ == "__main__":
    #  中心和宽高
    # testfile =r"E:\data1\archive\retail_data\val\images"
    # labelfile =r"E:\data1\archive\retail_data\val\labels"


    # testfile =r"E:\data1\train\from\archive\images"
    # labelfile =r"E:\data1\train\from\archive\labels"   #

    # testfile =r"E:\data1\train\from\retailproduct\val2019"
    # labelfile =r"E:\data1\train\from\retailproduct\val_labels"   #

    testfile =r"E:\data1\train\from\retailproduct\test2019"
    labelfile =r"E:\data1\train\from\retailproduct\test_labels"   #

    
    showimgfile(testfile,labelfile)
    