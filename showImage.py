# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  

from yaml import load
from model import model_mobilenetv2_seg_small as modellib
#from data.data_aug import Normalize_Img, Anti_Normalize_Img
#config_path = './config/model_mobilenetv2_pt.yaml'



config_path='./config/model_mobilenetv2_with_two_auxiliary_losses.yaml'
model_name = 'mobilenetv2_eg1800_with_two_auxiliary_losses.pth'




##########################
def Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)/scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = (img[j,:,:,i]-mean[i])*val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:,:,i] = (img[:,:,i]-mean[i])*val[i]
        return img

def Anti_Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = img[j,:,:,i]/val[i]+mean[i]
        return np.array(img*scale, np.uint8)
    else:
        for i in range(len(mean)):
            img[:,:,i] = img[:,:,i]/val[i]+mean[i]
        return np.array(img*scale, np.uint8)


def padding_img(img_ori, size=224, color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    img = np.zeros((max(height, width), max(height, width), 3)) + color
    
    if (height > width):
        padding = int((height-width)/2)
        img[:, padding:padding+width, :] = img_ori
    else:
        padding = int((width-height)/2)
        img[padding:padding+height, :, :] = img_ori
        
    img = np.uint8(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.array(img, dtype=np.float32)

def resize_padding(image, dstshape, padValue=0):
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

def generate_input(exp_args, inputs, prior=None):
    inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)
    
    if exp_args.video == True:
        if prior is None:
            prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
            inputs_norm = np.c_[inputs_norm, prior]
        else:
            prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
            inputs_norm = np.c_[inputs_norm, prior]
       
    inputs = np.transpose(inputs_norm, (2, 0, 1))
    return np.array(inputs, dtype=np.float32)

def pred_single(model, exp_args, img_ori, prior=None):
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    in_shape = img_ori.shape
    img, bbx = resize_padding(img_ori, [exp_args.input_height, exp_args.input_width], padValue=exp_args.padding_color)
    
    in_ = generate_input(exp_args, img, prior)
    in_ = in_[np.newaxis, :, :, :]
    
    if exp_args.addEdge == True:
        output_mask, output_edge = model(Variable(torch.from_numpy(in_)).cuda())
    else:
        output_mask = model(Variable(torch.from_numpy(in_)).cuda())
    prob = softmax(output_mask)
    pred = prob.data.cpu().numpy()
    
    predimg = pred[0].transpose((1,2,0))[:,:,1]
    out = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    out = cv2.resize(out, (in_shape[1], in_shape[0]))
    return out, predimg





####################
with open(config_path,'rb') as f:
    cont = f.read()
cf = load(cont)

exp_args = edict()    
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist'] # ['EG1800', ATR', 'MscocoBackground', 'supervisely_face_easy']

exp_args.model_root = cf['model_root'] 
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = cf['input_height']
# the width of input images, default=224
exp_args.input_width = cf['input_width']

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = cf['video']
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = cf['prior_prob']

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = cf['addEdge']
# whether to add consistency constraint loss, default=False
exp_args.stability = cf['stability']

# input normalization parameters
exp_args.padding_color = cf['padding_color']
exp_args.img_scale = cf['img_scale']
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = cf['img_mean']
# BGR order, image val, default=[0.017, 0.017, 0.017]
exp_args.img_val = cf['img_val'] 

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = cf['useUpsample'] 
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = cf['useDeconvGroup'] 

# print ('===========> loading model <===========')

netmodel_video = modellib.MobileNetV2(n_class=2, 
                                      useUpsample=exp_args.useUpsample, 
                                      useDeconvGroup=exp_args.useDeconvGroup, 
                                      addEdge=exp_args.addEdge, 
                                      channelRatio=1.0, 
                                      minChannel=16, 
                                      weightInit=True,
                                      video=exp_args.video).cuda()


bestModelFile = os.path.join(exp_args.model_root, model_name)

# from functools import partial
# import pickle
# pickle.load = partial(pickle.load, encoding="latin1")
# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
# checkpoint = torch.load(bestModelFile, map_location=lambda storage, loc: storage, pickle_module=pickle)


if os.path.isfile(bestModelFile):
    checkpoint_video = torch.load(bestModelFile)
    netmodel_video.load_state_dict(checkpoint_video['state_dict'])
    print ("minLoss: ", checkpoint_video['minLoss'], checkpoint_video['epoch'])
    print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint_video['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(bestModelFile))



###image
#wind
#img_ori = cv2.imread("//SmartGo-Nas/pentao/data/zch_edit/images/train/1/00003.png")
# mask_ori = cv2.imread("/home/dongx12/Data/EG1800/Labels/00457.png")

linuxpad = True
#linux 
img_ori = cv2.imread("/mnt/pentao/data/zch_edit/images/train/1/00011.png")
#img_ori = cv2.imread("/mnt/pentao/data/zch_edit/images/train/1/00003.png")

# img_path ='D:/pengt/data/mydata/val/images'

prior = None
height, width, _ = img_ori.shape

background = img_ori.copy()
background = cv2.blur(background, (17,17))

alphargb, pred = pred_single(netmodel_video, exp_args, img_ori, prior)

print("predict OK !!!")
if linuxpad:
    cv2.imwrite("./1_results.png",alphargb)
    np.save("alphargb.npy",alphargb)
    np.save("pred.npy",pred)
else:
    plt.imshow(alphargb)
    plt.show()
    print(alphargb.shape)

alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
result = np.uint8(img_ori * alphargb + background * (1-alphargb))

myImg = np.ones((height, width*2 + 20, 3)) * 255
myImg[:, :width, :] = img_ori
myImg[:, width+20:, :] = result
if linuxpad:
    np.save("showimg.npy",myImg[:,:,::-1]/255)
else:
    plt.imshow(myImg[:,:,::-1]/255)
    plt.yticks([])
    plt.xticks([])
    plt.show()




#video


# img_dir  = r'test_imgs'
# imgs = os.listdir(img_dir)
# for img in imgs:
#     img_name = img
#     # img_path  =  r'./test_imgs/00022.png'
#     img_path = os.path.join(img_dir,img)
#     print(img_path)
#     img = cv2.imread(img_path)
#     print(img.shape)
#     img = cv2.resize(img, (512, 512),
#                      interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
#     img1 = img.copy()
#     img = img.astype(np.float32)
#     # normalize the img (with the mean and std for the pretrained ResNet):
#     img = img / 255.0
#     # img = img - np.array([0.485, 0.456, 0.406])
#     # img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
#     img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))


#     # convert numpy -> torch:
#     img = torch.from_numpy(img)  # (shape: (3, 512, 1024))
#     img = torch.unsqueeze(img, 0)
#     img = img.cuda()
#     outputs = net(img)
#     # print(outputs.shape)
#     print(torch.unique(outputs))

#     # outputs = (outputs[0].detach().cpu() ).numpy()[0]#注意，这个东西是网络输出通道为1分类的时候的用法源码上是这么写的。

#     # outputs = np.where(outputs>0,255,0)
#     # outputs = outputs[0].detach().cpu().numpy()
#     # outputs = np.argmax(outputs,axis=0)

#     outputs = (outputs[0].data.cpu() > 0).numpy()[0]



#     print(outputs.shape)
#     # print(np.unique(outputs))


#     idx_fg = (outputs == 0)
#     syn_bg = cv2.imread(r'./1.jpg')
#     syn_bg = cv2.resize(syn_bg, (512, 512))
#     img_orig = cv2.resize(img1, (512, 512))

#     seg_img = 0 * img_orig
#     seg_img[:, :, 0] = img_orig[:, :, 0] * idx_fg + syn_bg[:, :, 0] * (1 - idx_fg)
#     seg_img[:, :, 1] = img_orig[:, :, 1] * idx_fg + syn_bg[:, :, 1] * (1 - idx_fg)
#     seg_img[:, :, 2] = img_orig[:, :, 2] * idx_fg + syn_bg[:, :, 2] * (1 - idx_fg)

#     # seg_img = cv2.resize(seg_img, (imgW, imgH))
#     outputs = np.where(outputs == 1, 0, 255)
#     outputs  = np.array(outputs,dtype=np.uint8)
#     result = 0 * img_orig
#     result[:, :, 0] = outputs
#     result[:, :, 1] = outputs
#     result[:, :, 2] = outputs
#     pic = np.concatenate((img1,seg_img,result),axis=1)
#     # cv2.imshow("2",seg_img)
#     # cv2.imshow("2", pic)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # cv2.imwrite("./result_imgs/{0}_results.png".format(img_name),pic)










