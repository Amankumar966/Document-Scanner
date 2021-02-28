# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:40:41 2020

@author: harshvardhan
"""


def CannyE(grayImageBlur):
    return cv2.Canny(grayImageBlur, 100, 300, 3)


def SobelE(img):
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    n,m = img.shape
    edges_img = img.copy()
    for row in range(3, n-2):
        for col in range(3, m-2):            
            local_pixels = img[row-1:row+2, col-1:col+2]
            vertical_transformed_pixels = vertical_filter*local_pixels
            vertical_score = vertical_transformed_pixels.sum()/4            
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = horizontal_transformed_pixels.sum()/4            
            edge_score = (vertical_score**2 + horizontal_score**2)**.5
            edges_img[row, col] = [edge_score]*3
    edges_img = edges_img/edges_img.max()
    return edges_img

import numpy as np 
import cv2
import imutils

img = cv2.imread("test.jpg")
orig = img.copy()

if img.shape[0]>1500 and img.shape[1]>1200:
    #Rescaling the image 
    scale_percent = 20# percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImageBlur = cv2.blur(grayImage,(3,3))



print("Which Edge Detection Technique :")
print("1 - Canny :")
print("2 - Sobel :")
n = int(input())
if n==1:
    edgedImage = CannyE(grayImageBlur)
if n==2:
    edgedImage = SobelE(grayImageBlur)

st = str(n)+"img.jpg"
cv2.imshow("Edge Detected Image", edgedImage)
cv2.imwrite(st, edgedImage)
cv2.waitKey(0) # press 0 to close all cv2 windows
cv2.destroyAllWindows()

# find the contours in the edged image, sort area wise 
# keeping only the largest ones 
allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
# descending sort contours area and keep top 1
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
# approximate the contour
perimeter = cv2.arcLength(allContours[0], True) 
ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
# show the contour on image
cv2.drawContours(img, [ROIdimensions], -1, (0,255,0), 2)
cv2.imshow("Contour Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# reshape coordinates array
ROIdimensions = ROIdimensions.reshape(4,2)
# list to hold ROI coordinates
rect = np.zeros((4,2), dtype="float32")
# top left corner will have the smallest sum, 
# bottom right corner will have the largest sum
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]
# top-right will have smallest difference
# botton left will have largest difference
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]
# top-left, top-right, bottom-right, bottom-left
(tl, tr, br, bl) = rect
# compute width of ROI

widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
maxWidth = max(int(widthA), int(widthB))
# compute height of ROI
heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
maxHeight = max(int(heightA), int(heightB))


# Set of destinations points for "birds eye view"
# dimension of the new image
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")
# compute the perspective transform matrix and then apply it
transformMatrix = cv2.getPerspectiveTransform(rect, dst)
# transform ROI
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
# lets see the wraped document
cv2.imshow("Scaned",scan)
cv2.waitKey(0)
cv2.destroyAllWindows()



# convert to gray
scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
# display final gray image
cv2.imshow("scanGray", scanGray)
cv2.waitKey(0)
cv2.destroyAllWindows()


from skimage.filters import threshold_local
# increase contrast incase its document
T = threshold_local(scanGray, 9, offset=8, method="gaussian")
scanBW = (scanGray > T).astype("uint8") * 255
# display final high-contrast image
cv2.imshow("scanBW", scanBW)
cv2.waitKey(0)
cv2.destroyAllWindows()