#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import os
import numpy as np
import cv2


#startNum = 1247   #   16   20180503
#startNum = 1264  #   38  C42  20180504
#startNum = 1301  #   31  Cx  20180504
#startNum = 1331  #   39  Cy  20180504
#startNum = 1370  #   42  Cz  20180504
#startNum = 1412  #   10 20180515
#startNum = 1422  #   16 20180522
#startNum = 1438  #    26 A 20180523
#startNum = 1464  #    37 b 20180523
#startNum = 1501 #    22 c 20180523
startNum = 1522 #    17 d 20180523
def autoMoveFile(pathA,pathB,path):#
    #light
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
            ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],
            ['19'],['20'],['21'],['22'],['23'],['24'],['25'],['26'],['27'],
            ['28'],['29'],['30'],['31'],['31'],['33'],['34'],['35'],['36'],
            ['37'],['38']] #20180504
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['13'],['15'],['16'],['17'],['18'],
        ['19'],['20'],['21'],['22'],['23'],['24'],['25'],['26'],['27'],
        ['28'],['29'],['30'],['31']] #20180504 cx
    '''
    #non-light     
    '''   
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],
        ['19'],['20'],['21'],['22'],['23'],['24'],['25'],['26'],['27'],
        ['28'],['29'],['30'],['31'],['32'],['33'],['34'],['35'],['36'],
        ['37'],['38'],['39'],['40'],['41'],['42']] #20180504
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10']] #20180515
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16']] #20180504 cx
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],
        ['19'],['20'],['21'],['22'],['23'],['24'],['25'],['26']] #20180523 a
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],
        ['19'],['20'],['21'],['22'],['23'],['24'],['25'],['26'],['27'],
        ['28'],['29'],['30'],['31'],['32'],['33'],['34'],['35'],['36'],
        ['37']] #20180523 b
    '''
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],
        ['19'],['20'],['21']] #20180523 c
    '''
    pairA=[['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],
        ['10'],['11'],['12'],['13'],['14'],['15'],['16'],['17']] #20180523 d
    
    path2 = ''
    strlit = 'l_nc_'
   # print(len(pairA),len(pairB))  
    for i in range(len(pairA)):
        #if 0:
        #i =0
        path0 = os.path.join(path,'%010d\\Glass'%(i+startNum))
        #print(path0)
        if not os.path.exists(path0):
            os.makedirs(path0)
        path1 = os.path.join(path,'%010d\\NonGlass'%(i+startNum))
        if not os.path.exists(path1):
            os.makedirs(path1)
        subPairAs = pairA[i]
        
        for subPairA in subPairAs:
            if subPairA == '_':
                continue
            if(i+1 != int(subPairA)):
                path0 = os.path.join(path,'%010d\\Glass'%(int(subPairA)-1+startNum))
                if not os.path.exists(path0):
                    os.makedirs(path0)
                path1 = os.path.join(path,'%010d\\NonGlass'%(int(subPairA)-1+startNum))
                if not os.path.exists(path1):
                    os.makedirs(path1)
            if (subPairA=='3'or subPairA=='11'):#or subPairA=='4'or subPairA=='6'or subPairA=='8'):# (subPairA=='6' or subPairA=='38' ): #4  # (subPairA=='19'):     ##########
                path2 = path0
            else:
                path2 = path1
            if 0: # subPairA =='12': # (subPairA=='1' or subPairA =='12'): #3
                strlit = 'nl_nc_'
            else:
                strlit = 'l_nc_'
            pathTmp = os.path.join(pathA,'0'+str(1000+i+1))
            fileList = [s for s in os.listdir(pathTmp) if 'feature' not in s and ('.bin' in s or '.jpg' in s)]
            for file in fileList:
                command = 'copy ' + os.path.join(pathTmp,file) + ' '+os.path.join(path2,strlit+subPairA+'_a_'+file)
                os.system(command)
                #print(command)
            #print(fileList)

    
        print('complete OK')
		
		
if __name__ == "__main__":
    oripathA = 'H:\\facedetection\\faceData\\originaldata\\20180523\\D\\register'
    oripathB = 'H:\\facedetection\\faceData\\20180411'
    dstpath = 'H:\\facedetection\\faceData\\originaldata\\OKdata'
    autoMoveFile(oripathA,oripathB,dstpath)