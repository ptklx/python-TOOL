import sys
import os
import numpy as np
import cv2
from shutil import copyfile
import shutil

def removefile(rootp):
    txtpath = os.path.join(rootp,'labels')
    imgpath = os.path.join(rootp,'images')
    backpath = os.path.join(rootp,'back')
    if not os.path.exists(backpath):
        os.makedirs(backpath, exist_ok=True)
    files = os.listdir(txtpath)
    for fi in files:
       
        lastpath = os.path.join(imgpath,fi.replace(".txt",'_0.jpg'))
        headpath = os.path.join(backpath,fi.replace(".txt",'_0.jpg'))
        # copyfile( lastpath,headpath)
        shutil.move(lastpath,headpath)


def renamefile(rootp):

    files = os.listdir(rootp)
    startn=1
    for fi in files:
        lastpath = os.path.join(rootp,fi)
        # headpath = os.path.join(rootp,fi.replace("WIN_20230808_13",''))
        headpath = os.path.join(rootp,f"{startn}.jpg")
        startn+=1
        os.rename(lastpath, headpath)


if __name__ == "__main__":
    # rootp=r"E:\data1\train\from\yolov5_data\train"
    rootp=r"E:\data1\train\from\yolov5_data\val"
    # removefile(rootp)

    ###rename
    # path = r"E:\data1\train\from\yolov5_data\train\back"
    # path = r"E:\data1\train\from\yolov5_data\val\back"
    path =r"E:\data1\our_collect\test_2img"
    renamefile(path)