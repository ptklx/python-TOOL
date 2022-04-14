import os

import cv2
import numpy as np


anchors_path  = 'model_data/seven/yolo_anchors_ori.txt'

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])


anchors_all = get_anchors(anchors_path)
anchors_mask = [[3,4,5],[1,2,3]]
img_size =[640,384]


anchors= np.reshape(anchors_all,[-1,2])[anchors_mask[0]]
stride_h = img_size[1] / 12
stride_w = img_size[0] / 20
#-------------------------------------------------#
#   此时获得的scaled_anchors大小是相对于特征层的
#-------------------------------------------------#
scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors]

print(scaled_anchors)



anchors= np.reshape(anchors_all,[-1,2])[anchors_mask[1]]
stride_h = img_size[1] / 24
stride_w = img_size[0] / 40
#-------------------------------------------------#
#   此时获得的scaled_anchors大小是相对于特征层的
#-------------------------------------------------#
scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors]

print(scaled_anchors)


#[(1.6875, 1.46875), (1.9375, 1.59375), (3.0625, 3.8125)]
# [(2.5625, 1.6875), (3.0, 2.4375), (3.375, 2.9375)]