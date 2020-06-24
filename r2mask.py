import cv2
import numpy as np

import os
import shutil

# img_dir = r'/mnt/zengch/extreme3cNet_data/train_data/leftImg8bit_trainvaltest/train/4'
masks_dir = r'D:/pengt/data/webvideo/zhoujielu/joiner_mask'
img_dir = 'D:/pengt/data/webvideo/zhoujielu/joinerpic'

out_dir = 'D:/pengt/data/webvideo/zhoujielu/clear'
imagespath = os.path.join(out_dir, 'images')
maskspath = os.path.join(out_dir, 'masks')

os.makedirs(imagespath, exist_ok=True)
os.makedirs(maskspath, exist_ok=True)


maskslist = os.listdir(masks_dir)
# name_imagelist = list(map(lambda x: os.path.splitext(x)[0], imagelist))
name_maskslist = list(map(lambda x: os.path.splitext(x)[0], maskslist))

for index, imgname in enumerate(name_maskslist):
    if os.path.exists(os.path.join(img_dir, imgname + ".jpg")):
        shutil.copy(os.path.join(img_dir, imgname + ".jpg"),imagespath)
        png = cv2.imread(os.path.join(masks_dir, imgname + ".png"), cv2.IMREAD_UNCHANGED)
        alpha = png[:, :, 3]
        alpha[alpha>0] = 200
        cv2.imwrite(os.path.join(maskspath, imgname + ".png"), alpha)
        print(index)