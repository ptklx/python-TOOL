import cv2
import os
import numpy as np
import random

def  main(img_path,back_path1,back_path2,outpath_images, out_path_back):

    if not os.path.exists(outpath_images):
        os.makedirs(outpath_images, exist_ok=True)
    if not os.path.exists(out_path_back):
        os.makedirs(out_path_back, exist_ok=True)  
    img_files = os.listdir(img_path)
    back_files1 = os.listdir(back_path1)
    back_files2 = os.listdir(back_path2)

    for fi in img_files:
        fi_path = os.path.join(img_path,fi)
        if os.path.isfile(fi_path):
            name,suffix= os.path.splitext(fi)
            if suffix ==".jpg" or suffix==".png":
                image1 = cv2.imread(fi_path)
                image1 = cv2.resize(image1,(int(image1.shape[1]/2),int(image1.shape[0]/2)))
                h,w,_ = image1.shape
                ###
                # bg = cv2.medianBlur(image1, 5)  # 对图像应用中值滤波
                # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                # blurred = cv2.GaussianBlur(image1, (5, 5), 0)
                # result = cv2.absdiff(image1, blurred)  # 计算图像与背景之间的差异
                # _, binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # # 应用自适应阈值分割
                # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                # edges = cv2.Canny(image1, 100, 200)
                # # _, binary = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # # 查找轮廓
                # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # # 在原始图像上绘制轮廓
                # cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)

                # # 使用掩膜进行图像分割
                # # result = cv2.bitwise_and(image1, binary)
                # cv2.imshow("test",image1)
                # cv2.waitKey(0)


                if random.random()>0.5:
                    back_i = random.choice(back_files1)
                    back_file_path = os.path.join(back_path1,back_i)
                    image2 = cv2.imread(back_file_path)
                    image2 = cv2.resize(image2,(w,h))
                    # result =cv2.addWeighted(image1,0.7,image2,0.3,0)
                    p = random.choice([0.4,0.5,0.6,0.7])
                    result =cv2.addWeighted(image1,p,image2,1-p,0)
                else:
                    back_i = random.choice(back_files2)
                    back_file_path = os.path.join(back_path2,back_i)
                    image2 = cv2.imread(back_file_path)
                    image2 = cv2.resize(image2,(w,h))
                    result = image1

                savepicname = os.path.join(outpath_images, fi)
                savepicnameback = os.path.join(out_path_back, fi)
                cv2.imwrite(savepicname, result)
                cv2.imwrite(savepicnameback, image2)
          
                # cv2.imshow("test",result)
                # cv2.waitKey(0)
                # 输出融合后的特征
                # print(fused_feature)

                # 使用融合后的特征进行进一步的处理或模型训练等

if __name__=="__main__":

    # img_path =r"E:\data1\train\from\retailproduct\val2019"
    # out_path =r"E:\data1\train\from\yolov5_data\val\images"
    #out_path_back =r"E:\data1\train\from\yolov5_data\val\back"

    img_path =r"E:\data1\train\from\retailproduct\test2019"
    out_path_imgs =r"E:\data1\train\from\yolov5_data\train\images"
    out_path_back =r"E:\data1\train\from\yolov5_data\train\back"

    back_path1 =r"E:\data1\train\from\yolov5_data\tmp\backrop"
    back_path2=r"E:\data1\train\from\yolov5_data\tmp\out"
    main(img_path,back_path1,back_path2,out_path_imgs,out_path_back)

