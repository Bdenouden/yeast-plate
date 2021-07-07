import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

imageList =  glob.glob(sys.path[0] + "/input/*.jpg")
[print(i) for i in imageList]

image = cv2.imread(imageList[1])

# convert image to RGB and HSV color space
img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# red lower mask (0-10)
lower_red = np.array([0,20,10])
upper_red = np.array([20,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# red upper mask (170-180)
lower_red = np.array([160,10,10])
upper_red = np.array([179,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join masks
mask_red = mask0+mask1

# white mask [h, s, b][0-179, 0-255, 0-255]
lower_white = np.array([0, 0, 120])
upper_white = np.array([179, 100, 255 ])
mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

# set output img to zero everywhere except my mask
img_out = image.copy()
img_out[np.where(mask_red==0)] = 0
img_out *= 255

# show red in different color
plt.imshow(img_out+img_RGB.copy())
plt.show()

# add edge to red blobs
edges = cv2.Canny(mask_red, threshold1=50, threshold2=60)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
img_out = img_RGB.copy()
img_out[ np.where(edges) ] = 255
plt.imshow(img_out)
plt.show()