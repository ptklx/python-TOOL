#!/usr/bin/python3
#!--*-- coding:utf-8 --*--
import cv2
import matplotlib.pyplot as plt

png = cv2.imread('/path/to/matting/matting_xxx.png', cv2.IMREAD_UNCHANGED)
print(png.shape)
#(800, 600, 4)

png_img = cv2.cvtColor(png[:,:,:3], cv2.COLOR_BGR2RGB)
alpha = png[:,:,3]

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(png_img)
plt.title("Matting PNG img")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(alpha)
plt.title("Matting Alpha img")
plt.axis("off")
plt.show()
