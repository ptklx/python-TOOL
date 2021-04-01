import numpy as np
import os
import cv2
import time
import argparse
import json
import sys
import xml.etree.ElementTree as ET


import platform
winNolinux = True
if platform.system().lower() == 'windows':
    winNolinux =True
    print("windows")
elif platform.system().lower() == 'linux':
    winNolinux = False
    print("linux")

#conda activate  pytorch16


def compute_overlap(a,b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    # ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
    # iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    # ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    iw = np.minimum(a[ 2], b[:, 2]) - np.maximum(a[ 0], b[:, 0])
    ih = np.minimum(a[ 3], b[:, 3]) - np.maximum(a[ 1], b[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua



def compute_overlap_diff(image_path,img,f_list, s_list):

    # np.expand_dims(d, axis=0)
        
    #for pa in range(len(f_list)):
    # x =np.array(f_list[pa])
    imgh, imgw ,_ = img.shape
    x= np.array(f_list)
    y = np.array(s_list)
    f_num = len(x)
    s_num = len(y)
    if f_num==0 and s_num==0:
        return
    if f_num!=s_num and (f_num==0 or s_num==0):
        for f1 in x:
            cv2.rectangle(img, (f1[0], f1[1]), (f1[2], f1[3]), (0, 0, 255), 1)  #red  x
            cv2.circle(img, (int((f1[2]+f1[0])/2), int((f1[3]+f1[1])/2)),6, (0, 0, 255), 1)
        for s1 in y:
            cv2.rectangle(img, (s1[0], s1[1]), (s1[2], s1[3]), (255, 0, 0), 2)   #blue y
        cv2.imshow("diff_num %s"%os.path.split(image_path)[1],img)
        cv2.waitKey(1000)
        return
    showFlag = False
    if f_num>s_num:
        for index ,fn in enumerate(x) :
            boxw = fn[2]-fn[0]
            boxh = (fn[3]-fn[1])  if(fn[3]-fn[1]) >0  else 0.00001
            ra = boxw/boxh
            if (fn[1]<10 or fn[3]>(imgh-10))and(ra>2.5 or ra<0.4):
                print("w and h  erro")
                cv2.rectangle(img, (fn[0], fn[1]), (fn[2], fn[3]), (0, 0, 255), 2)  #
                cv2.imshow("wandh erro",img)
                cv2.waitKey(100)
                continue
            overlaps=compute_overlap(fn,y)
            a_loc = np.argmax(overlaps, axis=0)
            if overlaps [a_loc]<0.5:
                showFlag = True
                #cv2.rectangle(img, (fn[0], fn[1]), (fn[2], fn[3]), (0, 0, 255), 1)  #red  x
                cv2.circle(img, (int((fn[2]+fn[0])/2), int((fn[3]+fn[1])/2)),6, (0, 0, 255), 2)
                #cv2.rectangle(img, (y[a_loc][0], y[a_loc][1]), (y[a_loc][2], y[a_loc][3]), (255, 0, 0), 2)  #blue  y
                cv2.circle(img, (int((y[a_loc][2]+y[a_loc][0])/2), int((y[a_loc][3]+y[a_loc][1])/2)),4, (255, 0, 0), 2)
    else:
        for index ,fn in enumerate(y) :
            boxw = fn[2]-fn[0]
            boxh = (fn[3]-fn[1])  if(fn[3]-fn[1]) >0  else 0.00001
            ra = boxw/boxh
            if (fn[1]<10 or fn[3]>(imgh-10))and(ra>2 or ra<0.5):
                print("w and h  erro")
                cv2.rectangle(img, (fn[0], fn[1]), (fn[2], fn[3]), (255, 0, 0), 2)  #
                cv2.imshow("wandh erro",img)
                cv2.waitKey(100)
                continue

            overlaps=compute_overlap(fn,x)
            a_loc = np.argmax(overlaps, axis=0)
            if overlaps [a_loc]<0.5:
                showFlag = True
                #cv2.rectangle(img, (fn[0], fn[1]), (fn[2], fn[3]), (255, 0, 0), 2)  #
                cv2.circle(img, (int((fn[2]+fn[0])/2), int((fn[3]+fn[1])/2)),6, (255, 0, 0), 2)
                #cv2.rectangle(img, (x[a_loc][0], x[a_loc][1]), (x[a_loc][2], x[a_loc][3]), (0, 0, 255), 1)
                cv2.circle(img, (int((x[a_loc][2]+x[a_loc][0])/2), int((x[a_loc][3]+x[a_loc][1])/2)),4, (0, 0, 255), 2)

            
    if  showFlag :
        img1 = img.copy()
        for f1 in x:
            cv2.rectangle(img1, (f1[0], f1[1]), (f1[2], f1[3]), (0, 0, 255), 1)  #red  x
            # cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        for s1 in y:
            cv2.rectangle(img, (s1[0], s1[1]), (s1[2], s1[3]), (255, 0, 0), 1)   #blue y
        cv2.imshow("result_overlap_blue2",img)
        cv2.imshow("result_overlap_red1",img1)
        cv2.waitKey(0)
    return 

def getboxslist(yolo_decoder,imgsrc,output_data,):

    boxes, probs = yolo_decoder.run(output_data,0.3)
    # boxes, probs = 0,0
    if len(boxes) == 0:
        return [],[]
    height, width = imgsrc.shape[:2]
    minmax_boxes = to_minmax(boxes)
    minmax_boxes[:,0] *= width
    minmax_boxes[:,2] *= width
    minmax_boxes[:,1] *= height
    minmax_boxes[:,3] *= height
    boxes = minmax_boxes.astype(np.int)
    centerpoint_list = []
    probs_list =[]
    for dex,b in enumerate(boxes):
        # print(probs[dex])
        if probs[dex]<0.5:
            continue
        centerpoint_list.append(b)
        probs_list.append(probs[dex])
    return centerpoint_list,probs_list



def xml2boxes(data_file,cropFlag=False):
    #annlists = os.listdir(data_file)
    dataSet = []
    #for fi in annlists:
    if os.path.splitext(data_file)[-1]!='.xml':
        return []
    #xml_path = os.path.join(data_file,fi)
    in_file = open(data_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls =="eggd":
            img = np.ones((100,100),dtype=np.uint8)
            cv2.imshow("eggd++++++++",img)
            cv2.waitKey(0)
        if cls not in ['chicken head','chicken head other','egg','egg half' ]:
            continue
        # cls_id =  0 #classes.index(cls)  ##############pt   label    0  
        
        xmlbox = obj.find('bndbox')
        if cropFlag:
            starty =int(xmlbox.find('ymin').text)-8 if (int(xmlbox.find('ymin').text)-8)>0 else 0
            endy = 223 if (int(xmlbox.find('ymax').text)-8)>223 else (int(xmlbox.find('ymax').text)- 8)
            b = [int(xmlbox.find('xmin').text),  starty,int(xmlbox.find('xmax').text),endy]
        else:
            b = [int(xmlbox.find('xmin').text),  int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text),int(xmlbox.find('ymax').text)]
        ####add
        if False:
            if((b[3]-b[2]) == 0 ):
                continue
            if( ((b[1]-b[0])/ (b[3]-b[2]))<1/2):
                continue
            if (((b[1]-b[0])/ (b[3]-b[2]))>2):
                continue
        #dataSet.append([b[1]-b[0], b[3]-b[2]])  #width height
        dataSet.append(b)
    in_file.close()
    #result = np.array(dataSet)
    return dataSet



if __name__ == '__main__':

    image_folds=r"Z:\data\train_egg\egg1m\train\from\test10"
    xml_folds=r"Z:\data\train_egg\egg1m\train\from\test10_xml"  #red 
    xml_folds_1 =r"Z:\data\train_egg\egg1m\train\from\test10_xml_0"  #blue


    piclist = os.listdir(image_folds)
    # piclist =  sorted(piclist,key= lambda x: int(x.split('.jp')[0])) 
    model_interpreter_time = 0
  
    for index, fi in enumerate(piclist):
        if os.path.splitext(fi)[1]!=".jpg": 
            continue
        # if index<343:
        #     continue
     
        image_path = os.path.join(image_folds,fi)
        xml_path = os.path.join(xml_folds, os.path.splitext(fi)[0]+".xml")   #red

        xml_path_1 = os.path.join(xml_folds_1, os.path.splitext(fi)[0]+".xml") #blue

        print(index,image_path)
        imgsrc = cv2.imread(image_path, cv2.IMREAD_COLOR)


        cropFlag = False
        imgh, imgw,_ = imgsrc.shape  
        if False: 
            if imgh == 240:
                imgsrc=imgsrc[8:232,:]
                cropFlag = True

            imgh, imgw,_ = imgsrc.shape
            if imgh!=224:
                cv2.imshow("not equal 224",imgsrc)
                cv2.waitKey(0)

        height, width = imgsrc.shape[:2]
        # imgsrc = cv2.resize(imgsrc, (320,224))
            # print()
        centerpoint_list0 = xml2boxes(xml_path,cropFlag)  # red

        centerpoint_list1 = xml2boxes(xml_path_1,cropFlag)  #  blue

        compute_overlap_diff(image_path,imgsrc,centerpoint_list0, centerpoint_list1)  #red blue
        # cv2.putText(imgsrc, "%f"%probs[dex], (b[0], b[1]), 0, 0.5, [100, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        # cv2.rectangle(imgsrc, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)

        if winNolinux:
            tl = 2 or round(0.002 * (imgsrc.shape[0] + imgsrc.shape[1]) / 2) + 1  # line/font thickness
            rightpoint=(0,height)  #local
            tf = max(tl - 1, 1) 
            # cv2.putText(imgsrc, "eggC:%d_%d_%d"%(egg_count,eggCount_big,eggCount_little), (rightpoint[0], rightpoint[1] - 2), 0, tl / 3, [255, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.imshow("result_OK",imgsrc)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print("over")
       
