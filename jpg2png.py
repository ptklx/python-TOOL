import cv2
import numpy as np


import os

# img_dir = r'/mnt/zengch/extreme3cNet_data/train_data/leftImg8bit_trainvaltest/train/4'
img_dir = r'./images_png'
dst_dir = r'./images'

imgs = os.listdir(img_dir)
print(len(imgs))
index =0
for img in imgs:
    img_name = img
    if os.path.splitext(img_name)[1]!='.png':
        continue
    
    img_path = os.path.join(img_dir,img)
    print(img_path)
    # print(len(image.shape))
    image = cv2.imread(img_path,cv2.IMREAD_COLOR)

    # print(len(image.shape))
    # if len(image.shape )==2:
    #     print("error")
    #     print(img_path)
    #     break
    # if len(image.shape )==3:
    #     pass
    #image = np.where(image > 127, 0,255)
    #print(np.unique(img))
    #image = np.array(image, dtype=np.uint8)
    # print(np.unique(image))
    #index+=1
    #print(index)
    #if index >10:
        #break
    
    cv2.imwrite(os.path.join(dst_dir,os.path.splitext(img_name)[0]+'.jpg'),image)
    