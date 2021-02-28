# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:52:05 2020

@author: harshvardhan
"""



import numpy as np 
import cv2

img = cv2.imread("test1.jpg")
if img.shape[0]>1500 and img.shape[1]>1200:

    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Resized image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(grayImage,(1,1),0)
gray3 = cv2.GaussianBlur(grayImage,(3,3),0)
gray5 = cv2.GaussianBlur(grayImage,(5,5),0)
gray7 = cv2.GaussianBlur(grayImage,(7,7),0)
gray1 = cv2.Canny(gray1,100,300)
gray3 = cv2.Canny(gray3,100,300)
gray5 = cv2.Canny(gray5,100,300)
gray7 = cv2.Canny(gray7,100,300)


cv2.imshow("1",gray1)
cv2.imshow("2",gray3)
cv2.imshow("3",gray5)
cv2.imshow("4",gray7)

cv2.imwrite("1c.jpg",gray1)
cv2.imwrite("2c.jpg",gray3)
cv2.imwrite("3c.jpg",gray5)
cv2.imwrite("4c.jpg",gray7)
cv2.waitKey(0)
cv2.destroyAllWindows()
