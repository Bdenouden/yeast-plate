import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import glob
from PIL import Image

"""
This script is intended to make the red groups stand out more.
Images that have to be processed should be placed in the input folder and have .jpg extension.
output will be multiple images in PDF format.

Author:
B. den Ouden
07-07-2021

"""

print("Starting image processing")

imagePathList =  glob.glob(sys.path[0] + "/input/*.jpg")
print("files found:")
[print(i) for i in imagePathList]
print()

def saveImgToPdf(pdfName, images = {}):
    arr = []
    for name, image in images.items():
        arr.append(Image.fromarray(image))

    arr[0].save(pdfName, save_all=True, append_images=arr[1:])

for num, imagePath in enumerate(imagePathList):

    print(f"Processing file {num+1}/{len(imagePathList)}")

    image = cv2.imread(imagePath)
    toPdf = {}
    pdfName = "output/"+os.path.basename(imagePath)+".pdf"

    # convert image to RGB and HSV color space
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    toPdf['original'] = img_RGB

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

    img_R2B = img_out+img_RGB.copy()
    toPdf["R2B"] = img_R2B

    # show red in different color
    # plt.imshow(img_R2B)
    # plt.show()

    # add edge to red blobs
    edges = cv2.Canny(mask_red, threshold1=50, threshold2=60)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_edges = img_RGB.copy()
    img_edges[ np.where(edges) ] = 255
    toPdf['edges'] = img_edges

    # plt.imshow(img_edges)
    # plt.show()

    saveImgToPdf(pdfName, toPdf)

print("All Done!")