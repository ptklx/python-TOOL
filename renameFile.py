#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import numpy as np
import cv2
from shutil import copyfile
import shutil

def renamefile(path,startNum,strend):
    files = os.listdir(path)
    # files =  sorted(files,key= lambda x: int(x.split('_')[0])) 
    
    count = 0
    
    for fi in files:
        fi_d = os.path.join(path,fi)
        if os.path.isfile(fi_d):
            # savel = fi.split('_')[1]
            name = os.path.splitext(fi)[0]
            suffix = os.path.splitext(fi)[1]
            if suffix ==".jpg":
                # savename = str(count)+'_'+strend+suffix
                # savename = str(count) + '_' + strend + suffix   ###add  a flag
                # savename = fi.replace("vid_36","vid_42")
                # savename = "led2_"+name+suffix
                # savename = "ql4_1_" + savel
                # savename = name+"dgd"+suffix
                savename = "zd2"+fi
                # savename =str(count)+suffix
                os.rename(fi_d,os.path.join(path,savename))
            if suffix ==".xml":
                # savename = "ledn_"+name+suffix
                savename = "zd2_"+fi
                # savename =str(count)+suffix
                os.rename(fi_d,os.path.join(path,savename))

            elif suffix ==".mp4":
                savename = str(count) + '_' + strend + suffix
                # savename = fi.replace("vid_36","vid_42")
                os.rename(fi_d, os.path.join(path, savename))

            count +=1
           

def copyfileF(filepath,sourcpath,target):
    # files = os.listdir(sourcpath)
    files = os.listdir(filepath)

    # for fi in files:
    #     headpath = os.path.join(sourcpath,fi.replace(".xml",'.jpg'))
    #     lastpath = os.path.join(target,fi.replace(".xml",'.jpg'))
    #     copyfile( headpath,lastpath)
    for fi in files:
        # headpath = os.path.join(sourcpath,fi.replace(".jpg",'.xml'))
        # lastpath = os.path.join(target,fi.replace(".jpg",'.xml'))
        headpath = os.path.join(sourcpath,fi)
        lastpath = os.path.join(target,fi)
        copyfile( headpath,lastpath)

def copyfileF1(sourcpath,target):
    files = os.listdir(sourcpath)

    for fi in files:
        lastpath = os.path.join(sourcpath,fi)
        headpath = os.path.join(target,"xj1301_"+fi)
        copyfile( lastpath,headpath)

def removefile(sourcpath, target):
    files = os.listdir(sourcpath)
    for fi in files:
        lastpath = os.path.join(target,fi.replace(".xml",'.jpg'))
        headpath = os.path.join(target+"_temp",fi.replace(".xml",'.jpg'))
        # copyfile( lastpath,headpath)
        shutil.move(lastpath,headpath)


if __name__ == "__main__":
    # testpath = r"E:\chicken_data\plant_pic\need-label\secondlabel\image"

    # testpath = r'Z:\data\train_egg\egg1m\train\from\image_new2\zd'
    # testpath = r'Z:\data\train_egg\egg1m\train\from\image_new4\2_xml'
    # renamefile(testpath,10,"_4")
    #remove
    # target = r'Z:\data\train_egg\egg1m\train\from\test12'
    # removefile(testpath, target)
    ###copy file to file
    # sourcpath  = r'Z:\data\train_egg\yolov5\temp2'
    # target = r"Z:\data\train_egg\yolov5\temp4"  
    # filepath  = r'Z:\data\train_egg\egg1m\train_all\from\overlap_images'

    # sourcpath = r"Z:\data\train_egg\egg1m\train_all\xml_i"
    # target = r"Z:\data\train_egg\egg1m\train_all\from\xml_overlap_single"  
    # copyfileF(filepath,sourcpath, target)

    # filepath = r"Z:\data\train_egg\egg1m\train_all\from\xml_overlap_test_correct"
    # target=r"Z:\data\train_egg\egg1m\train_all\from\image"
    # sourcePath=r"Z:\data\train_egg\egg1m\train_all\images"
    # copyfileF(filepath,sourcePath, target)
    filepath =r"Z:\data\train_egg\egg1m\train_all\xml_overlap"
 
    target=r"Z:\data\train_egg\egg1m\train_all\xml_i_overlap"
    sourcePath=r"Z:\data\train_egg\egg1m\train_all\xml_i"
    copyfileF(filepath,sourcePath, target)
    