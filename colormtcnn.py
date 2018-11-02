#!/usr/bin/python
# -*- coding: UTF-8 -*-

from scipy import misc
import sys

import os

import numpy as np
import math
import cv2

import random
#from time import sleep


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
def subListDir(filepath,steplength):
    path_list=[]
    files = os.listdir(filepath)
    files = sorted(files)
    if steplength > len(files):
        stepNum = len(files)
    else:
        stepNum = steplength
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            path_list += subListDir(fi_d,steplength)
        else:
            stepNum-=1
            if stepNum == 0:
                stepNum = steplength
                path_list.append(os.path.join(filepath,fi_d))
    return path_list


def parseFile(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    leftEye_dir = os.path.join(output_dir, 'leftEye')
    if not os.path.exists(leftEye_dir):
        os.makedirs(leftEye_dir)
    rightEye_dir = os.path.join(output_dir,'rightEye')
    if not os.path.exists(rightEye_dir):
        os.makedirs(rightEye_dir)

    with open(args.input_dir,'r') as f:
        for lines in f.readlines():
            line = lines.split()
            print(line[0])
def getoldLoc():
    f=open('./modleData/lastReadLoc.txt','r')
    strf =f.readline()
    listf =[]
    if strf != '':
        listf= strf.split('_')
    lenf = len(listf)
    f.close()
    if lenf>2:
        personCount = int(listf[1]) if int(listf[1])>0 else 0
        picCount = int(listf[2]) if int(listf[2])>0 else 0
        return personCount ,picCount
    elif lenf>1:
        personCount = int(listf[1]) if int(listf[1])>0 else 0
        return personCount , 0
    return 0 , 0
def setoldLoc(num,locnum):
    f= open('./modleData/lastReadLoc.txt','w')
    f.write(str(0)+'_'+str(num)+'_'+str(locnum))
    f.close()
class MtcnnDlib():
    def __init__(self, oripath ,subpath,coordinateL,coortxt):
        self.saveflag1 = 0
        self.saveflag2 = 0
        self.oriPath = oripath
        self.subpath = subpath
        self.detpath = subpath[2]
        self.coordinateL = coordinateL
        self.coordtxt = coortxt
        self.coordsize = len(coortxt)
        self.minsize = 20 # minimum size of face
        self.face = np.zeros(4, dtype=np.int32)
        self.eye = np.zeros (4, dtype=np.int32)
        
        self.datalist = []
    
    def readcoord(self):
        countNum =0
        path1 = []
        for ntxt in self.coordtxt:
            path1.append( self.coordinateL+'/'+ ntxt)
        for n in range(len(path1)):
            with open(path1[n],'r') as file_to_read:
                while True:
                    strline = file_to_read.readline()
                    if strline =='':
                        break
                    loc = strline.find(self.detpath)
                    movelen = len(self.detpath)
                    if loc == -1:               ######################only compute once
                        loc = strline.find(self.subpath[0])   #############
                        movelen = len(self.subpath[0])
                    deltstr = strline[loc+movelen:]
                    self.datalist.append(deltstr)
                    countNum+=1
                    
                    #if countNum>10:
                        #break
            file_to_read.close()
        return 
   
    def saveImage(self,hdp,positivePL,positivePR,negativeP):
        numeyeL = 0
        numeyeR = 0
        hdnum = 0
        negativeNum =0
        for n in range(len(self.datalist)):
            listf =[]
            if self.datalist[n] != '':
                listf= self.datalist[n].split()
            realpath = self.oriPath + listf[0]
            img = cv2.imread(realpath,cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE  cv2.IMREAD_COLOR
            if img is None:
                continue
            else:
                width = img.shape[0]
                height = img.shape[1]
                for m in range(4):
                    self.face[m] = int(listf[m+1])if int(listf[m+1])>0 else 0
                self.eye[0] = int(listf[5])if int(listf[5])>0 else 0
                self.eye[1] = int(listf[10])if int(listf[10])>0 else 0
                self.eye[2] = int(listf[6])if int(listf[6])>0 else 0
                self.eye[3] = int(listf[11])if int(listf[11])>0 else 0
                leneyex =  self.eye[2]- self.eye[0]
                #lenfacex = self.face[2]-self.face[0]
                lenfacey = self.face[3]-self.face[1]
            
                setplen = int(leneyex/3.6)

                startyL = self.eye[1]-setplen
                endyL = self.eye[1]+setplen
                startxL = self.eye[0]-setplen
                endxL = self.eye[0]+setplen

                startyR = self.eye[3]-setplen
                endyR = self.eye[3]+setplen
                startxR = self.eye[2]-setplen
                endxR = self.eye[2]+setplen
                
                if not(startyL < 0 or endyL > height-1 or startxL < 0 or endxL > width) \
                and not(startyL < 0 or endyL > height-1 or startxL < 0 or endxL > width):
                    '''
                    ImgL = img[startyL:endyL,startxL:endxL]
                    ImgR = img[startyR:endyR,startxR:endxR]
                    cv2.imwrite(positivePL + '/'+ str(numeyeL)+".jpg",ImgL)
                    cv2.imwrite(positivePR + '/'+ str(numeyeR)+".jpg",ImgR)
                    '''
                    numeyeL+=1
                    numeyeR+=1
                    ahd = random.randint(1,60)
                    flaghd = 0
                    if ahd < 0:
                        flaghd = 1
                    if flaghd:   #hd
                        flagLR =0
                        starty = startyL - 5
                        endy = endyL + 5
                        startx = startxL-5
                        endx = endxL + 5
                        if not(starty < 0 or endy>height-1 or startx<0 or endx >width-1):
                            img[starty:endy,startx:endx] = random.randint(1, 200)
                            flagLR +=1
                        starty = startyR - 5
                        endy = endyR + 5
                        startx = startxR-5
                        endx = endxR + 5
                        if not(starty < 0 or endy>height-1 or startx<0 or endx >width-1):
                            img[starty:endy,startx:endx] = random.randint(1, 200)
                            flagLR+=1
                        if flagLR==2:
                            cv2.imwrite(hdp + '/'+ str(hdnum)+".jpg",img)
                            hdnum+=1
                    movelen = setplen
                    borderLen = 2*setplen
                    countNumy = 0
                    countNumx = 0
                    an = random.randint(1,100)
                    b = 0
                    if an < 10:
                        b = 1
                    while b:   #negative
                        starty1 = startyL-movelen + movelen*countNumy
                        endy1 = starty1+borderLen
                        if endy1>height-1:
                            break
                        if endy1 >starty1+lenfacey+borderLen*2:
                            break

                        startx1 = movelen*countNumx
                        endx1= startx1+borderLen
                        countNumx+=1
                        nm = random.randint(1,60)
                        if endx1 >width-1:
                            countNumx=0
                            countNumy+=nm
                            continue
                        if abs(starty1+borderLen/2 - self.eye[1]) < borderLen*1.2 and abs(self.eye[0]-borderLen/2-startx1)<borderLen*1.2:
                            continue
                        if abs(starty1+borderLen/2 - self.eye[3]) < borderLen*1.2 and abs(self.eye[2]-borderLen/2-startx1)<borderLen*1.2:
                            continue
                        if not(starty1 < 0  or startx1<0 ):
                            negtiveImg = img[starty1:endy1,startx1:endx1]
                            cv2.imwrite(negativeP + '/'+ str(negativeNum)+".jpg",negtiveImg)
                            negativeNum+=1

                       
                   
                    

'''
    def saveFaceImage(self,flaghd,hdp,negativepath,inpic):
        

        eyedistance = self.eye_bb[1]- self.eye_bb[0]
        minx = min(self.eye_bb[1], self.eye_bb[0],self.eye_bb[2] ,self.eye_bb[3],self.eye_bb[4]) 
        maxx = max(self.eye_bb[1], self.eye_bb[0],self.eye_bb[2] ,self.eye_bb[3],self.eye_bb[4]) 

        miny = min(self.eye_bb[5],self.eye_bb[6],self.eye_bb[7], self.eye_bb[8], self.eye_bb[9]) 
        maxy = max(self.eye_bb[5],self.eye_bb[6],self.eye_bb[7], self.eye_bb[8], self.eye_bb[9]) 
        
        minx =  minx - int(eyedistance/2) if (minx  - eyedistance/2) > 0 else 0
        maxx =  maxx + int(eyedistance/2) if (maxx + int(eyedistance/2)<self.width-1) else self.width-1
        miny =  miny - int(eyedistance/2) if (miny - eyedistance/2) > 0 else 0
        maxy =  maxy + int(eyedistance/2) if (maxy + int(eyedistance)<self.height-1) else self.height -1
            
       
        setpwidth = 0
        if (maxy- miny)>(maxx-minx) and (maxx-minx)>0:
            setpwidth= int(((maxy- miny)+(maxx-minx)/4)/2)
        elif (maxy- miny)<(maxx-minx) and (maxy- miny)>0:
            setpwidth= int(((maxy- miny)/4+(maxx-minx))/2)
        else:
            return

        if flaghd:
            starty = miny
            endy = maxy
            startx = minx
            endx = maxx
            if not(starty < 0 or endy>self.height-1 or startx<0 or endx >self.width-1):
                MtcnnDlib.saveHdNum+=1
                inpic[starty:endy,startx:endx] = random.randint(1, 200)
                cv2.imwrite(hdp + '\\'+ str(MtcnnDlib.saveHdNum)+".bmp",inpic)
            return

        countNumy = 0
        countNumx = 0
        an = 1  #random.randint(1,8)
        b = 1
        if an < 3:
            b =1
        while b:
            starty1 = setpwidth*countNumy
            endy1 = starty1+maxy - miny
            if endy1>self.height-1:
                break
            startx1 = setpwidth*countNumx
            endx1= startx1+maxx - minx 
            countNumx+=1
            if endx1 >self.width-1:
                countNumx=0
                countNumy+=1
                continue
            if abs(starty1 - miny) < setpwidth and abs(startx1-minx)<setpwidth:
                continue
            if not(starty1 < 0  or startx1<0 ):
                MtcnnDlib.savenegativeNum+=1
                negtiveImg = inpic[starty1:endy1,startx1:endx1]
                cv2.imwrite(negativepath + '\\'+ str(MtcnnDlib.savenegativeNum)+".bmp",negtiveImg)
'''

    
    

if __name__ == '__main__':
    #mainx(parse_arguments(sys.argv[1:]))
    #V:\\pt_data   test   V:\\NIR_ALL
    #read faceLoc_x1 faceLoc_y1 faceLoc_x2 faceLoc_y2
    # eyeL_x  eyeR_x  nose_x  mouthL_x  mouthR_x    
    #  eyeL_y  eyeR_y  nose_y  mouthL_y  mouthR_y 
    coortxt= ['lfw_bounding_boxes_96529.txt','CASIA_bounding_boxes_17017.txt']
    headpath =['/home/deepmind/Dataset/','./datasets/','/datasets/']
    coordianeteL = 'H:/facedetection/faceData/colortxt'

    
    hdP = 'H:\\facedetection\\pic\\eyesSample\\coloreye\\eyehd'
    positivePL = 'H:\\facedetection\\pic\\eyesSample\\coloreye\\eyepositive\\lefteye'
    positivePR = 'H:\\facedetection\\pic\\eyesSample\\coloreye\\eyepositive\\righteye'
    negativeP = 'H:\\facedetection\\pic\\eyesSample\\grayeye\\eyenegative'

   
    
    tface = MtcnnDlib('V:/',headpath,coordianeteL,coortxt )
    tface.readcoord()
    tface.saveImage(hdP,positivePL,positivePR,negativeP)

    
    
   
  
    del tface
   
