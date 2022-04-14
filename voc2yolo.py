import xml.etree.ElementTree as ET
import os
import cv2
import random
#  from pathlib import Path
# classes = ['aeroplane', 'bicycle','bird','boat','bottle','bus','car', 'cat', 'chair','cow','diningtable','dog','horse','motorbike','person', 'pottedplant','sheep',
# 'sofa','train','tvmonitor']

import platform
winNolinux = True
if platform.system().lower() == 'windows':
    winNolinux =True
    print("windows")
elif platform.system().lower() == 'linux':
    winNolinux = False
    print("linux")
#主要是生成txt，等文件

# classes = ['chicken head','chicken head other' ]
classes = ['egg' ,'pig','overlap','overegg']
# dataflag = "rgb_01"
dataflag = ""

#xmlfolds = r'Z:\data\pig\pig_dete\Annotations'
# imgpath = "/home/pengtao/data/pig/pig_dete/JPEGImages"

# xmlfolds=r"Z:\data\train_egg\egg1m\train\xml"
# xmlfolds="/home/pengtao/data/train_egg/egg1m/train/xml"
# imgpath="/home/pengtao/data/train_egg/egg1m/train/images"
xmlfolds='/home/pengtao/data/train_egg/egg1m/train_overlap/xml_overlap'
imgpath="/home/pengtao/data/train_egg/egg1m/train_overlap/images"


# save_path = r'Z:\data\pig\pig_dete'
# save_path = r"Z:\data\train_egg\egg1m\train"
# save_path ="/home/pengtao/data/train_egg/egg1m/train"
save_path = "/home/pengtao/data/train_egg/egg1m/train_overlap"
if  winNolinux:
    part_name = 'trainpic_new_win.txt'  # test.txt
else:
    part_name = 'trainpic_new.txt'  # test.txt



txtsave_path = os.path.join(save_path,"labels_yolov4")
if not os.path.exists(txtsave_path):
        os.makedirs(txtsave_path)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)



def convert_annotation():
    # pic_file = open(os.path.join('E:/chicken_data/plant_pic/my_train/train_%s/'%dataflag ,part_name),'w')   ###change
    data_file = open(os.path.join(save_path ,part_name),'w')   ###change
 
    
    files = os.listdir(xmlfolds)
    for fi in files:
        if os.path.splitext(fi)[1]!=".xml":
            continue

        in_file = open(os.path.join(xmlfolds,fi),encoding='gb18030',errors='ignore')
        # print(fi)
    
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        imgw = float(size.find('width').text)
        imgh = float(size.find('height').text)
        #please create file
        label_file = open(os.path.join(txtsave_path,os.path.splitext(fi)[0]+'.txt'), 'w')  # 生成txt格式文件

        #label_file.write('E:/plant_pic/my_train/train/image/{}.jpg'.format(os.path.splitext(fi)[0]))
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id =  0 #classes.index(cls)  ##############pt   label    0  
            xmlbox = obj.find('bndbox')
            #b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text)]
            ######xyxy 2 center x center y w h
    
            # b =  ((b[0] + (b[1]-b[0])/2)/imgw,    (b[2] + (b[3]-b[2])/2)/imgh, (b[1]-b[0])/imgw, (b[3]-b[2])/imgh)
            b = [xmlbox.find('xmin').text,  xmlbox.find('ymin').text,xmlbox.find('xmax').text,xmlbox.find('ymax').text]
            if int(b[0])>int(b[2]) or int(b[1])>int(b[3]) :
                label_file.write(b[2] + " " + b[3] +" "+b[0]+" "+ b[1]+" "+ str(cls_id))
                print(fi)
            else:
                label_file.write(b[0] + " " + b[1] +" "+b[2]+" "+ b[3]+" "+ str(cls_id))
            label_file.write("\n")
        # pic_file.write(os.path.splitext(fi)[0])
      

        data_file.write('%s/{}.jpg'.format(os.path.splitext(fi)[0])%imgpath)
        data_file.write('\n')
    
        label_file.close()
    data_file.close()
 

def show():
    # cocopath = "E:/coco/coco/train2017.txt"
    cocopath = "E:/chicken_data/plant_pic/my_train/train/trainpic.txt"
    parent = os.path.split(cocopath)[0]+'/'   #str(Path(cocopath).parent) + os.sep
    t = open(cocopath, 'r')
    
    t = t.read().splitlines()
    x = t[3]   #########select img
    file_img = x.replace('./', parent) #if x.startswith('./')
    # f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
    labels_p = file_img.replace('images', 'labels').replace(os.path.splitext(file_img)[-1], '.txt')
    labels_t = open(labels_p,'r')
    img = cv2.imread(file_img)
    height ,width ,_ = img.shape

    color = [random.randint(0, 255) for _ in range(3)]
    labels_v = labels_t.readlines()
    for n in  range(len(labels_v)):
        x = [float(i) for i in labels_v[n].split()]
        #ceter xy  2 xyxy
        c1, c2 = (int((x[1]-x[3]/2)*width), int((x[2]-x[4]/2)*height)), (int((x[1]+x[3]/2)*width), int((x[2]+x[4]/2)*height))
        cv2.rectangle(img, c1, c2, color, 2)  # filled
        cv2.imshow("test",img)
        cv2.waitKey(0)
    cv2.imshow("test",img)
    cv2.waitKey(0)


 


if __name__ == "__main__":
   
    convert_annotation()
    # show()  #test