#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import numpy as np
import cv2

def renamefile(path,startNum):
    files = os.listdir(path)
    
    count = startNum
    
    for fi in files:
        fi_d = os.path.join(path,fi)
        if os.path.isfile(fi_d):
            savel = fi.split('_')
            savename = str(count+ int(savel[0]))+'_'+savel[1]+'_'+savel[2]
            os.rename(fi_d,os.path.join(path,savename))
            #count +=1
           

if __name__ == "__main__":
    testpath = 'H:\\facedetection\\faceData\\colorandinfr_'
    #testpath = '.\\negative'
    renamefile(testpath,200)