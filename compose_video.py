import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
import numpy as np
import cv2
import time
import onnxruntime as rt
from functools import reduce


##video
#vidpath = 'D:/pengt/segmetation/SHARE/huaibiao1.mp4'

video_path = '//SmartGo-Nas/pentao/data/video/111'
out_video_path = '//SmartGo-Nas/pentao/data/video/out_111'

# video_path = '/mnt/pentao/data/video/111'
# out_video_path = '/mnt/pentao/data/video/out_01'
#######onnx_path
ONNX_PATH='./4channels.onnx'

imgsize = (640, 384)   # width  height

usepic_or_model = False

mean = [0.5, 0.5, 0.5,0]  #0.5
std = [0.225, 0.225, 0.225,0.225]
mean = np.array(mean)
std = np.array(std)

maskvalue = 128


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class RKNNPredict(object):

    def __init__(self):

        self.sess = rt.InferenceSession(ONNX_PATH)
        self.premask = np.zeros((imgsize[1],imgsize[0]))

    def run(self, image,zeroflag = 1,index = 0):
        height, width = image.shape[0:2]
        bgr = cv2.resize(image, imgsize)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = np.expand_dims(self.premask, -1)
        img = np.concatenate((rgb, m), -1)
        img = img / 255
        img = img - mean
        img = img / std
        img = to_tensor(img)
        img1 = torch.from_numpy(img).unsqueeze(0)
        input_name = self.sess.get_inputs()[0].name
        out_name = self.sess.get_outputs()[0].name
        pred_onx = self.sess.run([out_name], {input_name: img1.cpu().detach().numpy()})
        out = pred_onx[0][0, 0, :, :]

        img_new_seg = (out > 0).astype(np.uint8)
        
        if index%20 == 0:
            self.premask = img_new_seg * 0   #
        else:
            self.premask = img_new_seg*maskvalue   #

        if zeroflag:
            self.premask = img_new_seg * 0

        img_new_seg = cv2.resize(img_new_seg.astype(np.uint8), (width,height), interpolation=cv2.INTER_LINEAR)
        return  img_new_seg



#video_name = r'huaibiao2.mp4'
# video_path = r'./videos/'+video_name
# videoCapture = cv2.VideoCapture(video_path)
# size = ((int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter = cv2.VideoWriter('result_kunkun.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

def resize_padding(image, dstshape, padValue=0):    # 等比例补边
    height, width, _ = image.shape
    ratio = float(width)/height # ratio = (width:height)
    dst_width = int(min(dstshape[1]*ratio, dstshape[0]))
    dst_height = int(min(dstshape[0]/ratio, dstshape[1]))
    origin = [int((dstshape[1] - dst_height)/2), int((dstshape[0] - dst_width)/2)]
    if len(image.shape)==3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape = (dstshape[1], dstshape[0], image.shape[2]), dtype = np.uint8) + padValue
        newimage[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
        newimage = np.zeros(shape = (dstshape[1], dstshape[0]), dtype = np.uint8)
        newimage[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    return newimage, bbx



class EvalSegIou(object):
    def __init__(self):
        if not usepic_or_model:
            self.pbpredict = RKNNPredict()
            self.pbpredictnext = RKNNPredict()
    def run(self):      
        if True :
            ve_list = os.listdir(video_path)
            n_b = 0 
            for v_name in ve_list:               
                videofi = os.path.join(video_path,v_name)
                avisuffix = os.path.splitext(v_name)[0]+".avi"
                outpath = os.path.join(out_video_path,avisuffix)
                print(n_b,"###")
                if n_b != 2 :      #select video
                    n_b+=1
                    continue
                n_b+=1
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
                out_height = 384
                out_width = 640

                out = cv2.VideoWriter(outpath,fourcc, 20.0, (out_width,out_height))

                cap=cv2.VideoCapture(videofi)
                index =0
                while (True):
                    ret,frame=cap.read()  
                    index+=1
                    startindex = 6000
                    if index <startindex:   
                        continue
                    if index>startindex+600:     # 6000
                        break
                    print(index)       
                    if ret == True:
                        if True:   #video

                            img_orig = cv2.resize(frame, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                            bg_mask = self.pbpredict.run(img_orig,0,index=index)

                            if True:
                                if bg_mask.sum()>4*4:
                                    index_t = np.argwhere(bg_mask > 0)

                                    row_h= [index_t[:,0].min(),index_t[:,0].max()]
                                    col_w= [index_t[:,1].min(),index_t[:,1].max()]
                                
                                    ori_row_h= np.dot(row_h,(frame.shape[0]/out_height)).astype(np.int32)
                                    ori_col_w= np.dot(col_w,(frame.shape[1]/out_width)).astype(np.int32)

                                    ori_h ,ori_w ,_=frame.shape
                                    gap_height = int((ori_row_h[1]-ori_row_h[0])/7)
                                    gap_width = int((ori_col_w[1]-ori_col_w[0])/4)
                                    dest_row_h=[0,ori_h]
                                    dest_col_w=[0,ori_w]
                                    dest_row_h[0] = 0 if (ori_row_h[0]-gap_height)<0 else ori_row_h[0]-gap_height
                                    dest_row_h[1] = ori_h if (ori_row_h[1]+gap_height)>ori_h else ori_row_h[1]+gap_height
                                    dest_col_w[0] = 0 if (ori_col_w[0]-gap_width)<0 else ori_col_w[0]-gap_width
                                    dest_col_w[1] = ori_w if (ori_col_w[1]+gap_width)>ori_w else ori_col_w[1]+gap_width

                                    # newimg = frame[ori_row_h[0]:ori_row_h[1],
                                    #              ori_col_w[0]:ori_col_w[1]]  # 裁剪坐标为[y0:y1, x0:x1]   人体
                                    
                                    newimg = frame[dest_row_h[0]:dest_row_h[1],
                                                 dest_col_w[0]:dest_col_w[1]]  # 裁剪坐标为[y0:y1, x0:x1]   人体


                                    #whole_man = cv2.resize(newimg, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                                    in_shape = newimg.shape
                                    in_img, bbx= resize_padding(newimg, [out_width,out_height])
                                    whole_bg_mask = self.pbpredictnext.run(in_img,0,index=index)
                                    out_get = whole_bg_mask[bbx[1]:bbx[3], bbx[0]:bbx[2]]
                                    out_get = cv2.resize(out_get, (in_shape[1], in_shape[0]))

                                    #tempf = np.copy( frame)
                                    big_f =np.zeros((frame.shape[0],frame.shape[1]), dtype = np.uint8) 

                                    # big_f[ori_row_h[0]:ori_row_h[1],ori_col_w[0]:ori_col_w[1]] [:] = out_get
                                    big_f[dest_row_h[0]:dest_row_h[1],dest_col_w[0]:dest_col_w[1]] [:] = out_get
                                    # temp=temp+whole_bg_mask
                                    #seg_imgnext = np.copy(frame)
                                    seg_imgnext=np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype = np.uint8) 
                                    seg_imgnext[:, :, 0] = frame[:, :, 0] * big_f 
                                    seg_imgnext[:, :, 1] = frame[:, :, 1] * big_f 
                                    seg_imgnext[:, :, 2] = frame[:, :, 2] * big_f 


                                    #cv2.imshow("mask",whole_bg_mask*200)
                                    cv2.imshow("big",seg_imgnext)
                                    cv2.imshow("test",newimg)
                                    cv2.waitKey(1)
                                


                            your_mask = bg_mask
                            #your_mask = cv2.resize(bg_mask, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                            
                            # cv2.imshow("test",bg_mask*200)
                            # cv2.waitKey(1)
                            your_mask = np.where(your_mask > 0, 1,0)
                            seg_img = 0 * img_orig
                            seg_img[:, :, 0] = img_orig[:, :, 0] * your_mask 
                            seg_img[:, :, 1] = img_orig[:, :, 1] * your_mask 
                            seg_img[:, :, 2] = img_orig[:, :, 2] * your_mask 
                            pic = np.array(seg_img,dtype=np.uint8)
                        #height, width = frame.shape[0:2]
                        else:
                            syn_bg = cv2.imread(imgpath)
                            syn_bg = cv2.resize(syn_bg.astype(np.uint8), (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                            img_orig = cv2.resize(frame, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                            
                            bg_mask = self.pbpredict.run(img_orig,0)
                            #your_mask = cv2.resize(bg_mask, (out_width,out_height), interpolation=cv2.INTER_LINEAR)
                            your_mask = bg_mask
                            seg_img = 0 * img_orig
                            seg_img[:, :, 0] = img_orig[:, :, 0] * your_mask + syn_bg[:, :, 0] * (1 - your_mask)
                            seg_img[:, :, 1] = img_orig[:, :, 1] * your_mask + syn_bg[:, :, 1] * (1 - your_mask)
                            seg_img[:, :, 2] = img_orig[:, :, 2] * your_mask + syn_bg[:, :, 2] * (1 - your_mask)

                            # seg_img = cv2.resize(seg_img, (imgW, imgH))
                            # outputs = np.where(your_mask == 1, 0, 255)
                            outputs = your_mask*255
                            outputs  = np.array(outputs,dtype=np.uint8)
                            result = 0 * img_orig
                            result[:, :, 0] = outputs
                            result[:, :, 1] = outputs
                            result[:, :, 2] = outputs
                            pic = np.concatenate((img_orig,result,seg_img),axis=1)
                    
                        # cv2.imshow("2",seg_img)
                        # cv2.imshow("3", pic)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                       
                        # cv2.imwrite("./result_img/%s.png"%index,pic)
                        out.write(pic)
                        # for i in range(height):
                        #     for j in range(width):
                        #         for k in range(3):
                        #             if your_mask[i,j]==1:
                        #                 bgrback[i][j][k]=frame[i][j][k]

                        # cv2.imshow("mask",your_mask*160)
                       
                        # cv2.imshow("video",bgrback)
                        # cv2.imshow("video",pic)
                        # if cv2.waitKey(1)&0xFF==ord('q'):
                            # break
                    else:
                        break

                cap.release()
        else:
            laststr = ''
            with open(dirrpath, 'r') as f:
                for line in f:
                    eline = line.strip()
                    name_tmp = dirr + 'images/' + eline+ '.jpg'
                    mask_tmp = dirr + 'masks/' + eline + '.png'
                    img = cv2.imread(name_tmp) #[...,::-1]
                    mask = cv2.imread(mask_tmp)[:,:,0]

                    if laststr != eline.split('_')[0]:
                        laststr = eline.split('_')[0]
                        zeroflag = 1
                    else:
                        zeroflag = 0
                    your_mask = self.pbpredict.run(img,zeroflag)
                    if True:
                        cv2.imshow("rgb",img)
                        cv2.imshow('dst', mask*160)
                        cv2.imshow("ori", your_mask*160)
                        cv2.waitKey(1)
               
        return 



def main():
    test = EvalSegIou()
    iou_all = test.run()
    print("OK")


if __name__ =="__main__":
    main()









