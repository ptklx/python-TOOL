import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
import numpy as np
import cv2
import math

def light_change(img):
    rows, cols = img.shape[:2]
    centerX = random.randint(int(rows/5),int(rows*4/5)) 
    centerY = random.randint(int(cols/5),int(cols*4/5)) 
    # 设置光照强度
    strength =random.randint(10,100) # 200

    radius = max(rows/2, cols/2)
   
    for i in range(rows):
        for j in range(cols):
            #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            #获取原始图像
            B =  img[i,j][0]
            G =  img[i,j][1]
            R = img[i,j][2]
            if (distance < radius*radius):
                #按照距离大小计算增强的光照值
                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result
                #判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                img[i,j] = np.uint8((B, G, R))
            else:
                img[i,j] = np.uint8((B, G, R))
    return img



def generate_img(path,size):

    ve_list = os.listdir(path)
    n_b = 0
    for v_name in ve_list:
        # if n_b ==0:
        #     n_b+=1
        #     continue
        # else:
        #     n_b+=1
        n_b+=1
        if os.path.splitext(v_name)[1]!='.jpg':
            continue
        img_path = os.path.join(path, v_name)
       
        savepicpath = os.path.join(path,"out")
        if not os.path.isdir(savepicpath):
            os.mkdir(savepicpath)

        index =0
        while (True):
            if index>200:
                break
            frame = cv2.imread(img_path)
            s = random.randint(size[0],size[1])
            if index%50!=0:
                frame =  light_change(frame)
            # cv2.imshow("test",frame)
            # cv2.waitKey(1)
            frame = cv2.resize(frame, (s, s))
            savepicname = os.path.join(savepicpath, '%d_%d.jpg'%(index,n_b))
            index += 1

            cv2.imwrite(savepicname, frame)
    



if __name__=="__main__":
    path = r"E:\data1\train\from\yolov5_data\tmp"
    size = [1700,1900]
    generate_img(path,size)

