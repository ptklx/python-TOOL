import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import cv2
import time
import random
#video_path = './'
#


if 1:
    # video_path = 'E:\data1\our_collect\high_speed2.mp4'
    # video_path=r"E:\data1\our_collect\high_speed3.mp4"
    # outpath = r'E:\data1\our_collect\high_speed3'
    video_path=r"E:\data1\our_collect\video_test\1\1.mp4"
    outpath = r'E:\data1\our_collect\video_test\1\1'
    if not os.path.isdir(outpath):
            os.mkdir(outpath)
    #savepicpath = os.path.join(video_path, suffix)
    cap = cv2.VideoCapture(video_path)
    index = 0
    allpic = 360
   # randnum=  random.randint(1,100)
    # stepnum = 7
    # nextsaveflag = 5000
    while (True):
        index += 1
        ret, frame = cap.read()
        if not ret:
            break
        # if index <2200:
        #     continue
        # if index>3000:
        #     break
        # if allpic<1:
        #     break
        # print(index)
        savepicname = os.path.join(outpath, '%d_m.jpg' % index)
        cv2.imwrite(savepicname, frame)
        # cv2.imshow("test",frame)
        # cv2.waitKey(5)
        # if nextsaveflag == index:
        #     allpic-=1
        #     print(index)
        #     savepicname = os.path.join(outpath, '%d_m.jpg' % index)
        #     cv2.imwrite(savepicname, frame)
        #     nextsaveflag = index + stepnum*randnum+ randnum
        # else:
        #     continue
    cap.release()

else:
    video_path = 'E:\data1\our_collect'
    ve_list = os.listdir(video_path)
    n_b = 0
    for v_name in ve_list:
        if os.path.splitext(v_name)[1]!='.mp4':
            continue
        videofi = os.path.join(video_path, v_name)
        n_b +=1
        suffix = os.path.splitext(v_name)[0]
        savepicpath = os.path.join(video_path,suffix)
        if not os.path.isdir(savepicpath):
            os.mkdir(savepicpath)

        cap = cv2.VideoCapture(videofi)
        index =0
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            savepicname = os.path.join(savepicpath, '%d_.jpg'%index)
            index += 1
            cv2.imwrite(savepicname, frame)
        cap.release()