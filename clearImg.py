#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os,shutil
import re
import numpy as np
import cv2




oirpath = '.\\negative'   
temppath = '.\\temp'
if not os.path.exists(temppath):                   #判断是否存在文件夹如果不存在则创建为文件夹  
    os.makedirs(temppath) 
files = os.listdir(oirpath)
numfiles = len(files)
#int(files[0][:-4])
rankfile = sorted(files,key = lambda x :  int(x.split('.bmp')[0]))

    
for i in range(numfiles-1):
    fi_d = os.path.join(oirpath,rankfile[i])
    oripic = cv2.imread(fi_d,cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    print(i)
    if oripic is  None:
        continue
    for  j in range(i+1,numfiles):
        fi_next = os.path.join(oirpath,rankfile[j])
        nextpic = cv2.imread(fi_next,cv2.IMREAD_GRAYSCALE)
        #print(j)
        if nextpic is  None:
            continue
        if(oripic.shape[0]!= nextpic.shape[0] or oripic.shape[1]!= nextpic.shape[1] ):
            continue
        diffValue = 0 
        diffmax = oripic.shape[0] *oripic.shape[1]/3
        for n in range(oripic.shape[0]):
            if diffValue > diffmax:
                break
            for m in range(oripic.shape[1]):
                if oripic[n][m] == 0  and  nextpic[n][m] == 0:
                    continue
                diffValue += abs((oripic[n][m]- nextpic[n][m]))/(oripic[n][m] + nextpic[n][m])
                if diffValue > diffmax:
                    break
        #cv2.imshow("one",oripic)
        #cv2.imshow("Two",nextpic)
        #cv2.waitKey(0)
        if diffValue < diffmax:
            #cv2.imshow("one",oripic)
            #cv2.imshow("Two",nextpic)
            #cv2.waitKey(0)
            dstpath = os.path.join(temppath,str(i)+'_'+ rankfile[j])
            shutil.move(fi_next,dstpath)
    



def showimg(inpathlist):
    inpathlist= 'V'+ inpathlist[1:]
    img = cv2.imread(inpathlist,cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    if img is None:
        return 0
    cv2.imshow("personOneFace",img)
    #cv2.waitKey(0)
    


   



