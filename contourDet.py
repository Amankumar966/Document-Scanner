# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:12:24 2020

@author: harshvardhan
"""


import numpy as np 
import cv2
import imutils

img = cv2.imread("test.jpg")
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImageBlur = cv2.GaussianBlur(grayImage,(3,3),0)
edgedImage = cv2.Canny(grayImageBlur, 100, 100, 3)


allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)
perimeter = cv2.arcLength(allContours[10], True) 
ROIdimensions = cv2.approxPolyDP(allContours[10], 0.02*perimeter, True)
cv2.drawContours(img, allContours, -1, (0,255,0), 2)

cv2.imshow("Contour Outline", img)
cv2.imwrite("AllContour.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
