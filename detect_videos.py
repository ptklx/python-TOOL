# camera-ready

# import sys
from models.extremeC3 import *
import torch.utils.data
import numpy as np
# import pickle
import matplotlib
matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import cv2
# import os

video_name = r'huaibiao2.mp4'
video_path = r'./test_videos/'+video_name
videoCapture = cv2.VideoCapture(video_path)
size = ((int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter = cv2.VideoWriter('result_kunkun.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
out = cv2.VideoWriter(r'./videos_results/'+video_name[:-4]+'.avi',fourcc, 20.0, (512*3,512))
net = ExtremeC3Net(classes=1, p=20, q=12)
# net = net.cuda()
# net_path = r'./parameters/ExtremeC3.pth'
net_path = r'./parameters/0420_epoch_24.pth'
net.load_state_dict(torch.load(net_path,map_location='cpu'))#
net.eval()#这个必须有。
while True:
	success, frame = videoCapture.read()
	# print(success)
	if success:
		img_size = 512
		# print(frame)
		frame_yuan = frame.copy()
		# image = frame[:,:,::-1]#bgr转换成rgb
		# image = Image.fromarray(image.astype('uint8')).convert('RGB')
		# img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
		# resize img without interpolation:
		img = frame.copy()

		img = cv2.resize(img, (img_size, img_size),
						 interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
		img1 = img[:,:,::-1].copy()
		# normalize the img (with the mean and std for the pretrained ResNet):

		# img = img - np.array([0.485, 0.456, 0.406])
		# img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
		img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
		img = img.astype(np.float32)

		# convert numpy -> torch:
		# img = torch.from_numpy(img).float().cuda()  # (shape: (3, 512, 1024))
		img = torch.from_numpy(img).float()  # (shape: (3, 512, 1024))
		img = img / 255.0
		img = torch.unsqueeze(img, 0)
		# print(img.shape)
		# network.eval()

		# image = Image.open(list_im[i])
		# print(image.shape)
		# print(type(image))
		# img = img.cuda()
		outputs = net(img)
		# print(outputs.shape)
		outputs = (outputs[0].detach().cpu() > 0).numpy()[0]
		# print(outputs.shape)
		# outputs = outputs[0].data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
		# outputs = np.argmax(outputs, axis=0)  # (shape: (batch_size, img_h, img_w))
		# outputs = outputs.astype(np.uint8)


		# print(pred_label_imgs.shape)  # 1,512,512
		pred_label_img = outputs # (shape: (img_h, img_w))
		idx_fg = (pred_label_img == 1)
		syn_bg = cv2.imread(r'./1.jpg')
		syn_bg = cv2.resize(syn_bg, (img_size,img_size))
		img_orig = cv2.resize(img1, (img_size, img_size))
		syn_bg = syn_bg[:, :, ::-1].copy()
		# print(idx_fg.shape,syn_bg.shape,img_orig.shape)
		seg_img = 0 * img_orig
		seg_img[:, :, 0] = img_orig[:, :, 0] * idx_fg + syn_bg[:, :, 0] * (1 - idx_fg)
		seg_img[:, :, 1] = img_orig[:, :, 1] * idx_fg + syn_bg[:, :, 1] * (1 - idx_fg)
		seg_img[:, :, 2] = img_orig[:, :, 2] * idx_fg + syn_bg[:, :, 2] * (1 - idx_fg)

		# seg_img = cv2.resize(seg_img, (imgW, imgH))
		outputs = np.where(outputs == 1, 255, 0)
		#
		outputs = np.array(outputs, dtype=np.uint8)

		result =  0 * img_orig
		result[:, :, 0] = outputs
		result[:, :, 1] = outputs
		result[:, :, 2] = outputs

		pic = np.concatenate((img1,result, seg_img), axis=1)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		pic = np.array(pic,dtype=np.uint8)
		pic =  cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
		cv2.imshow('frame',pic)
		out.write(pic)
	#
	else:
		break
videoCapture.release()  # 释放cap视频文件
cv2.destroyAllWindows()
