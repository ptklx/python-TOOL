#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import numpy as np
import cv2

colorNum = 0
infNum = 0
deep = 0
def autoMoveFile(pathA,pathB):#
    global colorNum
    global infNum
    global deep
    if not os.path.exists(pathB):
            os.makedirs(pathB)
    files = os.listdir(pathA)
    if files:
        if os.path.isdir(os.path.join(pathA,files[0])):
            if deep <3:
                if colorNum > infNum:
                    infNum = colorNum
                else: 
                    colorNum = infNum
            if deep == 2:
                sfiles = sorted(files,key = lambda item: item.split('_')[1] if item.find('color')!=-1 else item)
            else:
                sfiles = sorted(files)
            for fi in sfiles:
                fi_d = os.path.join(pathA,fi)
                deep+=1
                autoMoveFile(fi_d,pathB)
        else:
            sfiles = sorted(files,key = lambda item:int(item.split('_')[3]))
            for fi in sfiles:
                fi_d = os.path.join(pathA,fi)
                if os.path.isdir(fi_d):
                    deep+=1
                    autoMoveFile(fi_d,pathB)
                if os.path.isfile(fi_d):
                    if fi.find('colours') != -1:
                        savepath = os.path.join(pathB,str(colorNum)+'_color_')
                        if os.path.splitext(fi)[1] == '.bin':
                            (r,g,b) = read565rgb(fi_d, 240,320)
                            img = cv2.merge([b,g,r])
                            '''
                            data = np.fromfile(fi_d,np.uint8)
                            if len(data) != 153600:
                                print(fi_d)
                                continue
                           
                            img = data.reshape(320,240)
                            '''
                            cv2.imwrite(savepath+".bmp",img)
                        '''
                        if os.path.splitext(fi)[1] == '.bmp':
                            #os.rename(fi_d,savepath+".bin")
                            open(savepath+".bmp", "wb").write(open(fi_d, "rb").read()) 
                        '''
                        colorNum+=1
                    elif fi.find('brightness') != -1:
                        savepath = os.path.join(pathB,str(infNum)+'_inf_')
                        if os.path.splitext(fi)[1] == '.bin':
                            data = np.fromfile(fi_d,np.uint8)
                            if len(data) != 307200:
                                print(fi_d)
                                continue
                            img = data.reshape(640,480)
                            cv2.imwrite(savepath+".bmp",img)
                        '''
                        if os.path.splitext(fi)[1] == '.bmp':
                            #os.rename(fi_d,savepath+".bin")
                            open(savepath+".bmp", "wb").write(open(fi_d, "rb").read()) 
                        '''
                        infNum+=1
    deep-=1  
          
def readYuvFile(filename,width,height):
    fp=open(filename,'rb')
    uv_width=width//2
    uv_height=height//2
   
    Y=np.zeros((height,width),np.uint8,order='C')
    U=np.zeros((uv_height,uv_width),np.uint8,'C')
    V=np.zeros((uv_height,uv_width),np.uint8,'C')
    for m in range(height):
        for n in range(width):
            Y[m,n]=ord(fp.read(1))
    for m in range(uv_height):
        for n in range(uv_width):
            V[m,n]=ord(fp.read(1))
            U[m,n]=ord(fp.read(1))
    fp.close()
    return (Y,U,V)


def yuv2rgb(Y,U,V,width,height):
    U=np.repeat(U,2,0)
    U=np.repeat(U,2,1)
    V=np.repeat(V,2,0)
    V=np.repeat(V,2,1)
    rf=np.zeros((height,width),float,'C')
    gf=np.zeros((height,width),float,'C')
    bf=np.zeros((height,width),float,'C')
    rf=Y+1.14*(V-128.0)
    gf=Y-0.395*(U-128.0)-0.581*(V-128.0)
    bf=Y+2.032*(U-128.0)
    for m in range(height):
        for n in range(width):
            if(rf[m,n]>255):
                rf[m,n]=255
            if(gf[m,n]>255):
                gf[m,n]=255
            if(bf[m,n]>255):
                bf[m,n]=255
    r=rf.astype(np.uint8)
    g=gf.astype(np.uint8)
    b=bf.astype(np.uint8)
    return (r,g,b)

def  read565rgb(filename,width,height):
    re = 0xf8
    gr0 = 0xe0
    gr1 = 0x07
    bl = 0x1f
    fp=open(filename,'rb')
    R=np.zeros((height,width),np.uint8,order='C')
    G=np.zeros((height,width),np.uint8,'C')
    B=np.zeros((height,width),np.uint8,'C')
    for m in range(height):
        for n in range(width):
           data0 = ord(fp.read(1))
           data1 = ord(fp.read(1))
           B[m,n] = (data0&bl)<<3
           gn0 = (data0&gr0)>>3
           gn1 = (data1&gr1)<<5
           G[m,n]=gn0+gn1
           R[m,n]=(data1&re)
    fp.close()
    return (R,G,B)

if __name__ == "__main__":
    #needmove = 'H:\\facedetection\\faceData\\colorAndInf'
    needmove='I:\\HBB\\faceDetTest'
    tomove = 'H:\\facedetection\\faceData\\colorandinfr'
    autoMoveFile(needmove,tomove)

    #test= 'I:\\HBB\\faceDetTest\\00000003\\室内\\color_9\\ID_9_frame_1_colours.bin'

    #(r,g,b) = read565rgb(test, 240,320)
    #(y,u,v) = readYuvFile(test,240,320)
    #cv2.imshow('tde',y)
    #cv2.waitKey(0)
    #(r,g,b)=  yuv2rgb(y,u,v,240,320)
    '''
    #tesre ='I:\\HBB\\faceDetTest\\00000003\\室内\\color_9\\ID_9_frame_1_colours.bmp'
    tes ='H:\\facedetection\\faceData\colorandinfr\\ID_9_frame_1_colours.bmp'
    reaimg = cv2.imread(tes)
   
    b0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    g0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    r0 = np.zeros((reaimg.shape[0],reaimg.shape[1]),dtype=reaimg.dtype)
    b0[:,:] = reaimg[:,:,0]  # 复制 b 通道的数据
    g0[:,:] = reaimg[:,:,1]  # 复制 g 通道的数据
    r0[:,:] = reaimg[:,:,2]  # 复制 r 通道的数据
    cv2.imshow("Blue0",b0)
    cv2.imshow("Red0",r0)
    cv2.imshow("Green0",g0)
    
    img = cv2.merge([b,g,r])
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    
    cv2.imshow('ter',r)
    cv2.imshow('teb',b)
    cv2.imshow('teg',g)
    cv2.imshow("Blue", cv2.merge([b, g, r]))
    #cv2.imshow("Green", cv2.merge([ r,g,b]))
    #cv2.imshow("Red", cv2.merge([g,b, r]))
    #cv2.waitKey(0)
  
    #cv2.imshow('te',img)
    cv2.waitKey(0)
    '''
    




