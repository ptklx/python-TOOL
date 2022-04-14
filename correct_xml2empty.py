import numpy as np
import os
import cv2
import time
import argparse
import json
import sys
import xml.etree.ElementTree as ET

from label2xml  import  born_xml



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

#把有标签的清空
if __name__ == '__main__':
    xml_path = r"Z:\data\train_egg\egg1m\train\from\image_new5\hanwei1_temp"
    cropFlag =False
    centerpoint_list0 = xml2boxes(xml_path,cropFlag)  #
    out_xml_path =r"Z:\data\train_egg\egg1m\train\from\image_new5\hanwei1_temp2"
    label_list=[]
    xyxylist=[]
    image_path = r"Z:\data\train_egg\egg1m\train\from\image_new5\hanwei1"
    files = os.listdir(xml_path)
    for fi in files:
        lastpath = os.path.join(image_path,fi.replace(".xml",'.jpg'))
        imgsrc = cv2.imread(lastpath, cv2.IMREAD_COLOR)
        print(image_path)
        if imgsrc.shape[0]!=384:
            imgsrc = cv2.resize(imgsrc, (640,384))
        born_xml(lastpath, imgsrc, xyxylist, label_list,out_xml_path )