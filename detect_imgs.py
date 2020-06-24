import torch
import torch.nn as nn

'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import os

#注意，这里训练已经规定了显卡
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #用第一张显卡


from models.extremeC3 import *
from etc.help_function import *
from etc.utils import *

# 这个是关于整个模型的一个性能测试
#这个东西也需要注意。
# if test_config["loss"] == "Lovasz":  # 用这个 ，输出维度为1 的话用这个损失
#     test_config["num_classes"] = 1
#     print("Use Lovasz loss ")
#     Lovasz = True

# else:
#     print("Use Cross Entropy loss ")  #如果输出维度为2 的话用交叉熵损失。
#     Lovasz = False
#################### common model setting and opt setting  #######################################
net = ExtremeC3Net(classes=1, p=1, q=5)
net = net.cuda()
# net_path = r'./parameters/ExtremeC3.pth'
# net_path = r'./parameters/0417_epoch_7.pth'
# net.load_state_dict(torch.load(net_path))
# Max_name = test_config["weight_name"]



# if torch.cuda.device_count() > 0:#运行的这个
#     net.load_state_dict(torch.load(net_path))
#     # print(Max_name)
# else:
#     net.load_state_dict(torch.load(net_path, "cpu"))

use_cuda = torch.cuda.is_available()
# num_gpu = torch.cuda.device_count()

# if use_cuda:
#     # if num_gpu > 1:
#     #     model = torch.nn.DataParallel(model)
#     net = net.cuda()
img_dir  = r'test_imgs'
imgs = os.listdir(img_dir)
for img in imgs:
    img_name = img
    # img_path  =  r'./test_imgs/00022.png'
    img_path = os.path.join(img_dir,img)
    print(img_path)
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.resize(img, (512, 512),
                     interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
    img1 = img.copy()
    img = img.astype(np.float32)
    # normalize the img (with the mean and std for the pretrained ResNet):
    img = img / 255.0
    # img = img - np.array([0.485, 0.456, 0.406])
    # img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
    img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))


    # convert numpy -> torch:
    img = torch.from_numpy(img)  # (shape: (3, 512, 1024))
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    outputs = net(img)
    # print(outputs.shape)
    print(torch.unique(outputs))

    # outputs = (outputs[0].detach().cpu() ).numpy()[0]#注意，这个东西是网络输出通道为1分类的时候的用法源码上是这么写的。

    # outputs = np.where(outputs>0,255,0)
    # outputs = outputs[0].detach().cpu().numpy()
    # outputs = np.argmax(outputs,axis=0)

    outputs = (outputs[0].data.cpu() > 0).numpy()[0]

    print(outputs.shape)
    # print(np.unique(outputs))


    idx_fg = (outputs == 0)
    syn_bg = cv2.imread(r'./1.jpg')
    syn_bg = cv2.resize(syn_bg, (512, 512))
    img_orig = cv2.resize(img1, (512, 512))

    seg_img = 0 * img_orig
    seg_img[:, :, 0] = img_orig[:, :, 0] * idx_fg + syn_bg[:, :, 0] * (1 - idx_fg)
    seg_img[:, :, 1] = img_orig[:, :, 1] * idx_fg + syn_bg[:, :, 1] * (1 - idx_fg)
    seg_img[:, :, 2] = img_orig[:, :, 2] * idx_fg + syn_bg[:, :, 2] * (1 - idx_fg)


    # seg_img = cv2.resize(seg_img, (imgW, imgH))
    outputs = np.where(outputs == 1, 0, 255)
    outputs  = np.array(outputs,dtype=np.uint8)
    result = 0 * img_orig
    result[:, :, 0] = outputs
    result[:, :, 1] = outputs
    result[:, :, 2] = outputs
    pic = np.concatenate((img1,seg_img,result),axis=1)
    # cv2.imshow("2",seg_img)
    # cv2.imshow("2", pic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./result_imgs/{0}_results.png".format(img_name),pic)










