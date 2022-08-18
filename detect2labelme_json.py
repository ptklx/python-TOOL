#coding: utf-8
import os
import json
import cv2
import numpy as np 
###这里是把文件读取转换成labelme的json形式

def convertlist2json(outputdir,image_path,boxlist,boxes_classes_list):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    out_json_file = os.path.join(outputdir,os.path.splitext(os.path.split(image_path)[1])[0]+".json")
    face_point_name=("leye","reye","nose","lmouth","rmouth")
    shapes =[]
    for index,b in enumerate(boxlist):
        if not isinstance(boxlist[0],list):
            b = b.tolist()
        data1 = dict(
            label=boxes_classes_list[index],
            points=[[float(b[0]),float(b[1])],[float(b[2]),float(b[3])]],
            shape_type="rectangle",
            flags={},
            group_id=None,
        )
        shapes.append(data1)
        if len(b)>14:
            for index,p in enumerate(face_point_name):
                data2 = dict(
                    label=p,
                    points=[[float(b[5+index*2]),float(b[6+index*2])]],
                    shape_type="point",
                    flags={},
                    group_id=None,
                )
                shapes.append(data2)
    
    imageHeight =0
    imageWidth =0
    if  os.path.isfile(image_path):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        imageHeight,imageWidth,_=img_raw.shape
    data = dict(
        version="5.0.0",
        flags={},
        shapes=shapes,
        imagePath=image_path,
        imageData=None,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )
    
    try:
        with open(out_json_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise 


def draw_show(img_raw,dets):
    for b in dets:
        if b[4] < 0.6:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    cv2.imshow("detect face",img_raw)


def main(main_path,out_json_fold):
    pic_folder = os.path.join(main_path, 'images')

    label_folder = os.path.join(main_path, 'label')
    list_name = os.listdir(pic_folder)

    # list_imgname = sorted(list_imgname)  #不改变原序列
    # list_name = sorted(list_name,key=lambda x:int(x.split(".jp")[0])) 
    # out_json_fold=r"D:\data\face\single_face_1\train_whole\evaluation\json"
    out_json_fold=out_json_fold
    for img_p in list_name:
        img_path = os.path.join(pic_folder,img_p)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label_file =os.path.join(label_folder,img_p.replace(".jpg",".txt")) 
        boxs =[]
        
        with open(label_file,'r') as f:
            content_txt = f.readlines()
            box_num = int(content_txt[1].strip())
            
            for k in range(box_num):
                boxs.append(content_txt[k+2].strip().split())
                
        boxs = np.array(boxs, dtype=np.float32)

#######################
        box_list =[]
        boxs_class_list =[]
        for b in boxs:
            if b[4] < 0.5:
                continue
            box_list.append(b)
            boxs_class_list.append("face")
        convertlist2json(out_json_fold,img_path,box_list,boxs_class_list)
################################
        draw_show(img_raw,boxs)
        cv2.waitKey(1)


if __name__=="__main__":
    # main_path=r"D:\data\face\single_face_1\train_whole\evaluation"
    # out_json_fold=r"D:\data\face\single_face_1\train_whole\evaluation\json"

    main_path=r"D:\data\face\single_face_1\train_whole\train"
    out_json_fold=r"D:\data\face\single_face_1\train_whole\train\json"

    main(main_path,out_json_fold)