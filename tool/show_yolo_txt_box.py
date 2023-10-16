import sys
import os
import numpy as np
import cv2
from shutil import copyfile

from PIL import Image, ImageDraw, ImageFont
# import shutil
import math
import re
import json
import codecs

cart_category = ["上好佳鲜虾40g/袋","乐事无限翡翠黄瓜味薯片罐装40克","雪碧清爽柠檬味汽水200ml","可口可乐芬达迷你罐200m","统一阿萨姆青提茉莉奶茶450ml/瓶",\
                 "农夫山泉550ml","哇哈哈无汽苏打水350ml"," 洁柔face软抽150抽","洁柔face古龙香水可湿水面纸抽纸","康师傅红烧牛肉超爽方便面143g",\
                 "财友素食(口水鸡)","优多邦维维熊抽纸","乐事酸辣柠檬凤爪味薯片","美好甜玉米火腿肠30g","可口可乐碳酸饮料汽水","维达杀菌无刺激湿纸巾80抽",\
                 "心相印牌纸面巾无香100抽"," 大黑桶","怡宝矿泉水"]


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_path = r'D:\algorithm\yolov5\tool\simfang.ttf'
    fontStyle = ImageFont.truetype(
        font_path, textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




def showimgfile(testfile,input_txt_path):
    ve_list = os.listdir(testfile)
    n_b = 0
    for v_name in ve_list:
        if os.path.splitext(v_name)[1] not in (".jpg",".bmp",".jpeg"):
            continue
        imgfi = os.path.join(testfile, v_name)

        txt_path = os.path.join(input_txt_path,  os.path.splitext(v_name)[0]+".txt")
        if not os.path.exists(txt_path):
            print("the label not exists %s",txt_path)
            continue
        img = cv2.imread(imgfi)
        print(imgfi)
        h,w,_ = img.shape
        # with open(input_txt_path, 'r', encoding='utf8') as fp:
        with open(txt_path, 'r') as file:
            lb = np.array([x.split() for x in file.read().strip().splitlines()], dtype=np.float32)  # labels
            box_info=lb
            for box_in in box_info:
                label = int(box_in[0])
                box = box_in[1:5]
                x,y,width,height=box
                x = x*w
                y = y*h
                width= width*w
                height = height*h
                startx = int(x-width/2)
                starty = int(y-height/2)
                endx = int(x+width/2)
                endy = int(y+height/2)

                # 设置字体  
                
      
                text = cart_category[label]
          
                color = (255, 0, 0)  # 字体颜色，红色  
                t_size = 30
                img = cv2AddChineseText(img,text,(startx, starty-t_size),color,textSize=t_size)
                cv2.rectangle(img,(startx,starty),(endx,endy),(0,255,0),2)

        img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
        cv2.imshow("test",img)
        cv2.waitKey(0)


        

if __name__ == "__main__":


    testfile =r"E:\data1\train\from\yolov5_cart\all_img"
    input_txt_path =r"E:\data1\train\from\yolov5_cart\test_labels_19"   #


    showimgfile(testfile,input_txt_path)
    