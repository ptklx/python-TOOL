import cv2
import numpy as np

import os

# img_dir = r'/mnt/zengch/extreme3cNet_data/train_data/leftImg8bit_trainvaltest/train/4'
#img_dir = r'/mnt/zengch/extreme3cNet_data/train_data/gtFine/train/6'
# img_dir = '//SmartGo-Nas/pentao/data/portrait/GT_png'
# dst_dir = '//SmartGo-Nas/pentao/data/portrait/masks'
img_dir='D:\\pengt\\data\\mydata\\train\\masks'
dst_dir = 'D:\\pengt\\data\\mydata\\train\\masks_255'
imgs = os.listdir(img_dir)
print(len(imgs))
for img in imgs:
    img_name = img

    img_path = os.path.join(img_dir,img)
    print(img_path)
    # print(len(image.shape))
    image = cv2.imread(img_path,0)

    # print(len(image.shape))
    # if len(image.shape )==2:
    #     print("error")
    #     print(img_path)
    #     break
    # if len(image.shape )==3:
    #     pass
    image = np.where(image > 0, 255,0)
    print(np.unique(img))
    image = np.array(image, dtype=np.uint8)
    # print(np.unique(image))
    cv2.imwrite(os.path.join(dst_dir,img_name),image)