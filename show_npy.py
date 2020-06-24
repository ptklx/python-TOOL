import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import cv2

alpha_path = '//SmartGo-Nas/pentao/code/sinet_all/PortraitNet-master/alphargb.npy'
pre_path = '//SmartGo-Nas/pentao/code/sinet_all/PortraitNet-master/pred.npy'
npy_path = '//SmartGo-Nas/pentao/code/sinet_all/PortraitNet-master/showimg.npy'


al = np.load(alpha_path)
plt.imshow(al)
plt.show()
pr = np.load(pre_path)
result = pr>0.5
result = np.array(result*200,dtype=np.uint8)
cv2.imshow("pre",result)
cv2.waitKey(0)
#plt.imshow(pr)
#plt.show()
b = np.load(npy_path)
plt.imshow(b)
plt.yticks([])
plt.xticks([])
plt.show()