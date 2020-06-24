import cv2
import numpy as np


import os
import shutil
# img_dir = r'/mnt/zengch/extreme3cNet_data/train_data/leftImg8bit_trainvaltest/train/4'
img_dir = r'//SmartGo-Nas/pentao/data/mydata/test'
imagespath = os.path.join(img_dir, 'images')
maskspath = os.path.join(img_dir, 'masks')
imagelist = os.listdir(imagespath)
maskslist =os.listdir(maskspath)
name_imagelist = list(map(lambda x: os.path.splitext(x)[0],imagelist))
name_maskslist = list(map(lambda x: os.path.splitext(x)[0],maskslist))

for inedex , imgname in enumerate(name_imagelist):
    if imgname not in name_maskslist:   #.find('a')
        img_path =os.path.join(imagespath,imgname+'.jpg')
        dst_path = os.path.join("../",imgname+".jpg")
        shutil.move(img_path,dst_path)
        print("pic_name:%s"%imgname)
    
for inedex , imgname in enumerate(name_maskslist):
    if imgname not in name_imagelist:   #.find('a')
        img_path =os.path.join(maskspath,imgname+".png")
        dst_path = os.path.join("../",imgname+'.png')
        shutil.move(img_path,dst_path)
        print("mask_name:%s" % imgname)



    # img_path = os.path.join(imagespath,imgname)
    # print(img_path)
    # # print(len(image.shape))
    # image = cv2.imread(img_path,-1)

    # # print(len(image.shape))
    # # if len(image.shape )==2:
    # #     print("error")
    # #     print(img_path)
    # #     break
    # # if len(image.shape )==3:
    # #     pass
    # image = np.where(image > 127, 0,255)
    # print(np.unique(img))
    # image = np.array(image, dtype=np.uint8)
    # # print(np.unique(image))
    # cv2.imwrite(os.path.join(img_dir,img_name),image)