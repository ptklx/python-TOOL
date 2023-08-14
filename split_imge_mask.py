import cv2
import os
import re
import numpy as np




def getRoiImage(image, roi):
    """
    获取图片ROI区域
    """
    if len(roi) == 0:
        return image
    img_h, img_w = image.shape[:2]
    x, y, w, h = roi

    int_roi =np.int0([x * img_w, y * img_h, w * img_w, h * img_h])
    x, y, w, h = int_roi

    return image[y:y+h, x:x+w]

class ImagePrepTool:
    """
    图片预处理工具类
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.w_num, self.h_num = tuple(cfg.cropNumXY)
        self.resizeSize = tuple(map(lambda x, y: x * y, cfg.modelSize, cfg.cropNumXY))

    def getImageList(self, raw_img):
        """
        1. Crop ROI
        2. Resize Data (raw & label)
        3. Crop Image Patch (by cropNumXY)
        """
        roi_img = getRoiImage(raw_img, self.cfg.roi)

        resize_img = cv2.resize(roi_img, self.resizeSize, interpolation=cv2.INTER_NEAREST)


        row_list = [np.split(split_w, self.w_num, axis=1) for split_w in np.split(resize_img, self.h_num, axis=0)]
        img_list = [block for row in row_list for block in row]

        return img_list
    
def slice_image_with_labels(image, labels, slice_size):
    height, width, _ = image.shape
    sliced_images = []
    sliced_labels = []
    
    for y in range(0, height, slice_size):
        for x in range(0, width, slice_size):
            if (y+slice_size)>height or (x+slice_size)>width:  #不切宽高小于slice_size 的
                break
            # 切片标签
            slice_labels = []
            for label in labels:
                if label_in_slice(label, x, y, slice_size):
                    # 根据切片位置调整边界框坐标
                    adjust_label(label, x, y, slice_size)
                    # 转yolo格式
                    label[0] = label[0]+label[2]/2
                    label[1] = label[1]+label[3]/2
                    new_list =[item/slice_size for item in label]
                    slice_labels.append(new_list)
           
            if len(slice_labels)<1:  #没有标签的不切
                continue
            #切图像
            slice_img = image[y:y+slice_size, x:x+slice_size]
            sliced_images.append(slice_img)
            sliced_labels.append(slice_labels)
    
    return sliced_images, sliced_labels

def label_in_slice(label, x, y, slice_size):
    # 检查边界框是否在切片内
    label_x, label_y, label_width, label_height = label
    return (label_x >= x) and (label_x + label_width <= x + slice_size) and \
           (label_y >= y) and (label_y + label_height <= y + slice_size)

def label_in_slice2(label, x, y, slice_size):  #优化切的方法
    # 检查边界框是否在切片内
    label_x, label_y, label_width, label_height = label

    return (label_x >= x) and (label_x + label_width <= x + slice_size) and \
           (label_y >= y) and (label_y + label_height <= y + slice_size)


def adjust_label(label, x, y, slice_size):
    # 根据切片位置调整边界框坐标
    label[0] -= x
    label[1] -= y




def main(testfile,labelfile,outpic_fold):
    slice_size = 640    ####################################
    files = os.listdir(testfile)
    # files =  sorted(files,key= lambda x: int(x.split('_')[0])) 
    outpic_images_path = os.path.join(outpic_fold,"images")
    if not os.path.exists(outpic_images_path):
        os.makedirs(outpic_images_path, exist_ok=True)
    outpic_labels_path = os.path.join(outpic_fold,"labels")
    if not os.path.exists(outpic_labels_path):
        os.makedirs(outpic_labels_path, exist_ok=True)

    for fi in files:
        fi_d = os.path.join(testfile,fi)
        if os.path.isfile(fi_d):
            # savel = fi.split('_')[1]
            name = os.path.splitext(fi)[0]
            suffix = os.path.splitext(fi)[1]
            if suffix ==".jpg" or suffix==".png":
                img = cv2.imread(fi_d)
                img = cv2.resize(img,(int(img.shape[1]),int(img.shape[0])))
                h,w,_ = img.shape
                label_txt = os.path.join(labelfile,name+".txt")
                boxes = []
                with open(label_txt,'r') as f:
                    txt_lines = f.readlines()
                    for i in txt_lines:
                        box_str = re.split('[\t \n]',i.strip())
                        box = list(map(float, box_str))
                        boxes.append(box)
                # print(boxes)
                labels =[]
                for box in boxes:
                    _,stx,sty,ex,ey=box
                    cenx =stx*w
                    ceny =sty*h
                    boxw = ex*w
                    boxy = ey*h
                    startx = int(cenx-boxw/2)
                    starty = int(ceny-boxy/2)

                    # cv2.rectangle(img,(startx,starty),(startx+int(boxw),starty+int(boxy)),(0,255,0),2)
                    # label_str = str(int(label) )
                    labels.append([startx,starty,boxw,boxy])

                sliced_images, sliced_labels = slice_image_with_labels(img, labels, slice_size)
                for i, (sliced_image, slice_labels) in enumerate(zip(sliced_images, sliced_labels)):
                    savepicname = os.path.join(outpic_images_path, name+'_%d.jpg' % i)
                    savelabelsname = os.path.join(outpic_labels_path, name+'_%d.txt' % i)
                    cv2.imwrite(savepicname, sliced_image)
                    with open(savelabelsname,'w') as f:
                        for label in slice_labels:
                            line = ' '.join(str(round(coord,6))for coord in label)
                            f.write('0 '+ line +'\n')
                        


                # cv2.imshow("test",img)
                # cv2.waitKey(0)


if __name__=="__main__":
    testfile =r"E:\data1\archive\retail_data\val\images"
    labelfile =r"E:\data1\archive\retail_data\val\labels"
    outpic_fold =r"E:\data1\train\from"
    name="archive"
    outpic_fold_f = os.path.join(outpic_fold,name)
    if not os.path.exists(outpic_fold_f):
        os.makedirs(outpic_fold_f, exist_ok=True)

    main(testfile,labelfile,outpic_fold_f)



    # # 读取图像和标签
    # 'E:\data1\archive\retail_data\test'
    # image = cv2.imread('image.jpg')
    # labels = [[100, 120, 50, 60], [200, 180, 70, 80], [350, 250, 30, 40]]  #xmin, ymin, w,h

    # # 定义切片大小
    # slice_size = 640

    # # 进行数据切片和标签切片
    # sliced_images, sliced_labels = slice_image_with_labels(image, labels, slice_size)

    # # 显示切片结果和标签
    # for i, (sliced_image, slice_labels) in enumerate(zip(sliced_images, sliced_labels)):
    #     cv2.imshow(f'Slice {i}', sliced_image)
    #     print(f'Slice {i} labels:', slice_labels)
        
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()