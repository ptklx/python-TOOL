#-*- coding: UTF-8 -*-  
 
from PIL import Image
 
def addTransparency(img, factor = 0.7 ):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0,0,0,0))
    img = Image.blend(img_blender, img, factor)
    return img
 
 
img = Image.open( "SMILEY.png ")
img = addTransparency(img, factor =0.7)



from PIL import Image
 
img = Image.open("SMILEY.png ")
img = img.convert('RGBA')
r, g, b, alpha = img.split()
alpha = alpha.point(lambda i: i>0 and 178)
img.putalpha(alpha)





from matplotlib import pyplot as plt 
import numpy as np 
import cv2 

img = cv2.imread('image.jpg') 

mask = np.zeros(img.shape[:2], np.uint8) 
bgdModel = np.zeros((1,65), np.float64) 
fgdModel = np.zeros((1,65), np.float64) 
rect = (50, 50, 450, 290) 

# Grabcut 
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT) 

r_channel, g_channel, b_channel = cv2.split(img) 
a_channel = np.where((mask==2)|(mask==0), 0, 255).astype('uint8') 

img_RGBA = cv2.merge((r_channel, g_channel, b_channel, a_channel)) 
cv2.imwrite("test.png", img_RGBA) 

# Now for plot correct colors : 
img_BGRA = cv2.merge((b_channel, g_channel, r_channel, a_channel)) 

plt.imshow(img_BGRA), plt.colorbar(),plt.show()


import cv2
import numpy as np
 
img = cv2.imread("/home/shuai/Desktop/lena.jpg")
 
b_channel, g_channel, r_channel = cv2.split(img)
 
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
# 最小值为0
alpha_channel[:, :int(b_channel.shape[0] / 2)] = 100
 
img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
 
cv2.imwrite("lena.png", img_BGRA)
