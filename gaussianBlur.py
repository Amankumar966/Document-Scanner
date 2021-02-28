# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:48:50 2020

@author: harshvardhan
"""

import cv2 
import numpy as np 

img = cv2.imread("test.jpg")
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayImageBlur =cv2.GaussianBlur(grayImage,(5,5),10000)
cv2.imshow("blur",grayImageBlur)
cv2.imwrite("res3.jpg",grayImageBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()