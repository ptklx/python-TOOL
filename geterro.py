#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import math
import cv2

def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())
# euclidean,欧式距离算法，传入参数为两个向量，返回值为欧式距离

def Manhattan(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()
# Manhattan_Distance,曼哈顿距离

def Chebyshev(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return max(np.abs(npvec1-npvec2))
# Chebyshev_Distance,切比雪夫距离


def Mahalanobis(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    npvec = np.array([npvec1, npvec2])
    sub = npvec.T[0]-npvec.T[1]
    inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))
# MahalanobisDistance,马氏距离



def Edit_distance_array(str_ary1, str_ary2):
    len_str_ary1 = len(str_ary1) + 1
    len_str_ary2 = len(str_ary2) + 1
    matrix = [0 for n in range(len_str_ary1 * len_str_ary2)]
    for i in range(len_str_ary1):
        matrix[i] = i
    for j in range(0, len(matrix), len_str_ary1):
        if j % len_str_ary1 == 0:
            matrix[j] = j // len_str_ary1
    for i in range(1, len_str_ary1):
        for j in range(1, len_str_ary2):
            if str_ary1[i-1] == str_ary2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len_str_ary1+i] = min(matrix[(j-1)*len_str_ary1+i]+1, matrix[j*len_str_ary1+(i-1)]+1, matrix[(j-1)*len_str_ary1+(i-1)] + cost)
    distance = int(matrix[-1])
    similarity = 1-int(matrix[-1])/max(len(str_ary1), len(str_ary2))
    return {'Distance': distance, 'Similarity': similarity}


def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))

class Getmeanerror():
    
    def __init__(self,inpath,inpath1,flag):
        if(flag==1):
            self.txtpathlist = self.subListDir(inpath,1)
            self.Numtxt = len(self.txtpathlist)
        else:
            self.txtpathlist = inpath
            self.Numtxt = 1
        self.AllimgNum = 0
        self.errorleftdiff = 0
        self.errorrightdiff = 0
        self.imagepath=''

    def subListDir(self ,filepath,steplength):
        path_list=[]
        files = os.listdir(filepath)
        files = sorted(files)
        stepNum = steplength
        for fi in files:
            fi_d = os.path.join(filepath,fi)
            if os.path.isdir(fi_d):
                path_list += self.subListDir(fi_d,steplength)
            else:
                stepNum-=1
                if stepNum == 0:
                    stepNum = steplength
                    path_list.append(os.path.join(filepath,fi_d))
        return path_list
    def readtxtcontent(self,path):
        f=open(path,'r')
        num = 0
        errorleftdiff = 0
        errorrightdiff = 0
        for line in f:
            #strf =f.readline()
            listf =[]
            listf= line.split(' ')
            self.imagepath =listf[0]
            lenf = len(listf)
            if lenf>18:
                intlist = []
                for j in listf[1:lenf]:
                    intlist.append(int(j))
                if intlist[14] != -1 and intlist[1] != -1:   #   not detecteye
                    num+=1
                    left ,right =self.computeDiff(intlist,lenf-1)
                    errorleftdiff += left
                    errorrightdiff += right
        f.close()
        return num, errorleftdiff,  errorrightdiff
    def computeDiff(self, listdata,lent):
        value1 = Euclidean(listdata[4:6],listdata[14:16])  #left
        value2 = Euclidean(listdata[6:8],listdata[16:18])  #right
        distva = Euclidean(listdata[4:6],listdata[6:8])
        divalue1 = value1/distva
        divalue2 = value2/distva
        if divalue1>2 or divalue2>2:
            img = cv2.imread(self.imagepath,cv2.IMREAD_COLOR)
            imgt = cv2.resize(img,(480,572))
            cv2.circle(imgt,(listdata[4],listdata[5]),2,(0,0,255),-1)
            cv2.circle(imgt,(listdata[6],listdata[7]),2,(0,0,255),-1)
            cv2.circle(imgt,(listdata[14],listdata[15]),2,(255,0,255),-1)
            cv2.circle(imgt,(listdata[16],listdata[17]),2,(255,0,255),-1)
            cv2.imshow('img',imgt)
            cv2.waitKey(0)

        #value = value1/distva +value2/distva
        return  divalue1, divalue2
    def computemuch(self,inNum):
        if(self.Numtxt == 1):
            gNum , diffleft ,diffright = self.readtxtcontent(self.txtpathlist)
            self.AllimgNum += gNum
            self.errorleftdiff += diffleft
            self.errorrightdiff += diffright
        else:
            for i in range(self.Numtxt):
                if self.AllimgNum>inNum:
                    break
                gNum , diffleft ,diffright = self.readtxtcontent(self.txtpathlist[i])
                self.AllimgNum += gNum
                self.errorleftdiff += diffleft
                self.errorrightdiff += diffright
    
    def GetAllerrorValue(self,var1 , var2):
        leftfeyeerror = self.errorleftdiff /self.AllimgNum 
        righteyeerro = self.errorrightdiff/self.AllimgNum 
        print("num image %d\n"%self.AllimgNum)
        print("leftfeyeerror:%f\n"%leftfeyeerror)
        print("rightfeyeerror:%f\n"%righteyeerro)
        




if __name__ == '__main__':
    #path = "./npdgood/1" 
    path = "V:/NIR_ALLlabeltxt/npdfaceseven/npdgood/chgcombine1.txt" 
    path2 = 'V:/NIR_ALLlabeltxt/npdfaceseven/npdgood/1'
    flag = 0
    Num = 1000
    clValue = Getmeanerror(path,path2, flag)
    clValue.computemuch(Num)
    clValue.GetAllerrorValue(0,0)


