'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-06-12 09:44:19
LastEditors: xiaoshuyui
LastEditTime: 2021-01-05 10:19:45
'''
try:
    from labelme import __version__ as labelme_version
except:
    labelme_version = '4.2.9'

# from convertmask import baseDecorate
import sys

sys.path.append("..")

import copy
import json
import os

import cv2
import numpy as np
import skimage.io as io
import yaml
# from convertmask.utils.img2xml.processor_multiObj import img2xml_multiobj
# from convertmask.utils.methods import rmQ
# from convertmask.utils.methods.getShape import *
# from convertmask.utils.methods.img2base64 import imgEncode
# import warnings
# from convertmask.utils.methods.logger import logger


def rs(st:str):
    s = st.replace('\n','').strip()
    return s


def readYmal(filepath, labeledImg=None):
    if os.path.exists(filepath):
        if filepath.endswith('.yaml'):
            f = open(filepath)
            y = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
            # print(y)
            tmp = y['label_names']
            objs = zip(tmp.keys(), tmp.values())
            return sorted(objs)
        elif filepath.endswith('.txt'):
            f = open(filepath,'r',encoding='utf-8')
            classList = f.readlines()
            f.close()
            l3 = [rs(i) for i in classList]
            l = list(range(1,len(classList)+1))
            objs = zip(l3,l)
            return sorted(objs)
    elif labeledImg is not None and filepath == "":
        """
        should make sure your label is correct!!!
        """
        labeledImg = np.array(labeledImg, dtype=np.uint8)

        labeledImg[labeledImg > 0] = 255
        labeledImg[labeledImg != 255] = 0

        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            labeledImg)

        labels = np.max(labels) + 1
        labels = [x for x in range(1, labels)]

        classes = []
        for i in range(0, len(labels)):
            classes.append("class{}".format(i))

        return zip(classes, labels)
    else:
        raise FileExistsError('file not found')



currentCV_version = cv2.__version__  #str
def get_approx(img, contour, length_p=0.1):
    """获取逼近多边形

    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()
    # 逼近长度计算
    epsilon = length_p * cv2.arcLength(contour, True)
    # 获取逼近多边形
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx


def getMultiRegion(img, img_bin):
    """
    for multiple objs in same class
    """
    # tmp = currentCV_version.split('.')
    if float(currentCV_version[0:3]) < 3.5:
        img_bin, contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # print(len(contours))
    regions = []
    if len(contours) >= 1:
        # region = get_approx(img, contours[0], 0.002)
        # return region
        # elif len(contours)>1:
        for i in range(0, len(contours)):
            if i != []:
                # print(len(contours[i]))
                region = get_approx(img, contours[i], 0.002)
                # print(region)
                if region.shape[0] > 3:
                    regions.append(region)

        return regions
    else:
        return []

def getBinary(img_or_path, minConnectedArea=20):
    if isinstance(img_or_path, str):
        i = cv2.imread(img_or_path)
    elif isinstance(img_or_path, np.ndarray):
        i = img_or_path
    else:
        raise TypeError('Input type error')

    if len(i.shape) == 3:
        img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    else:
        img_gray = i

    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5,5),np.uint8)
    # img_bin = cv2.dilate(img_bin,kernel)
    # img_bin = cv2.erode(img_bin,kernel)

    # img_bin[img_bin!=0] = 255

    # img_bin = morphology.remove_small_objects(img_bin,3)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
    # print(stats.shape)
    for index in range(1, stats.shape[0]):
        if stats[index][4] < minConnectedArea or stats[index][4] < 0.1 * (
                stats[index][2] * stats[index][3]):
            labels[labels == index] = 0

    labels[labels != 0] = 1

    img_bin = np.array(img_bin * labels).astype(np.uint8)
    # print(img_bin.shape)

    return i, img_bin



def process(oriImg):
    img, img_bin = getBinary(oriImg)

    return getMultiRegion(img, img_bin)


def rmqrm(filepath):
    p = open(filepath, 'r+')

    lines = p.readlines()

    d = ""
    for line in lines:
        c = line.replace('"group_id": "null",', '"group_id": null,')
        d += c

    p.seek(0)
    p.truncate()
    p.write(d)
    p.close()

def getMultiShapes(oriImgPath,
                   labelPath,
                   savePath='',
                   labelYamlPath='',
                   flag=False,
                   areaThresh=500):
    """
    oriImgPath : for change img to base64  \n
    labelPath : after fcn/unet or other machine learning objects outlining , the generated label img
                or labelme labeled imgs(after json files converted to mask files)  \n
    savePath : json file save path  \n
    labelYamlPath : after json files converted to mask files. if doesn't have this file,should have a labeled img.
                    but the classes should change by yourself(labelme 4.2.9 has a bug,when change the label there will be an error.
                    )   \n

    """
    # print('-==================')
    # print(oriImgPath)
    # print(labelPath)
    # print(savePath)
    # print(labelYamlPath)
    # print('-==================')
    if isinstance(labelPath, str):
        if os.path.exists(labelPath):
            label_img = io.imread(labelPath)
        else:
            raise FileNotFoundError('mask/labeled image not found')
    else:
        # img = oriImg
        label_img = labelPath
    
    # print(np.max(label_img))

    if np.max(label_img) > 127:
        # print('too many classes! \n maybe binary?')
        label_img[label_img > 127] = 255
        label_img[label_img != 255] = 0
        label_img = label_img / 255

    labelShape = label_img.shape

    labels = readYmal(labelYamlPath, label_img)
    # print(list(labels))
    shapes = []
    obj = dict()
    obj['version'] = labelme_version
    obj['flags'] = {}
    for la in list(labels):

        if la[1] > 0:
            # print(la[0])
            img = copy.deepcopy(label_img)   # img = label_img.copy()
            img = img.astype(np.uint8)

            img[img == la[1]] = 255

            img[img != 255] = 0

            region = process(img.astype(np.uint8))

            if isinstance(region, np.ndarray):
                points = []
                for i in range(0, region.shape[0]):
                    # print(region[i][0])
                    points.append(region[i][0].tolist())
                shape = dict()
                shape['label'] = la[0]
                shape['points'] = points
                shape['group_id'] = 'null'
                shape['shape_type'] = 'polygon'
                shape['flags'] = {}
                shapes.append(shape)

            elif isinstance(region, list):
                # print(len(region))
                for subregion in region:
                    points = []
                    for i in range(0, subregion.shape[0]):
                        points.append(subregion[i][0].tolist())
                    shape = dict()
                    shape['label'] = la[0]
                    shape['points'] = points
                    shape['group_id'] = 'null'
                    shape['shape_type'] = 'polygon'
                    shape['flags'] = {}
                    shapes.append(shape)

    # print(len(shapes))
    obj['shapes'] = shapes
    # print(shapes)
    (_,imgname) = os.path.split(oriImgPath)
    obj['imagePath'] = imgname   #这个文件名很重要，是通过这个访问
    # print(obj['imagePath'])
    # obj['imageData'] = str(imgEncode(oriImgPath))
    obj['imageData'] = None #str(imgEncode(oriImgPath))

    obj['imageHeight'] = labelShape[0]
    obj['imageWidth'] = labelShape[1]

    j = json.dumps(obj, sort_keys=True, indent=4)

    

    # print(j)

    if not flag:
        saveJsonPath = savePath + os.sep + obj['imagePath'][:-4] + '.json'
        # print(saveJsonPath)
        with open(saveJsonPath, 'w') as f:
            f.write(j)

        rmqrm(saveJsonPath)

    else:
        return j



if __name__ == "__main__":
    # test()
    imgPath = './tools/multi_objs_wrap.jpg'
    maskPath = './tools/label_255.png'
    savePath = './tools'

    getMultiShapes(imgPath, maskPath, savePath) 
