# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:53:43 2020

@author: harshvardhan
"""

import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('test.jpg')

# Apply gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply gaussian blur
blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

# Positive Laplacian Operator
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

cv2.imshow(laplacian)

plt.figure()
plt.title('Shapes')
plt.imsave('shapes-lap.jpg', laplacian, cmap='gray', format='png')
plt.imshow(laplacian, cmap='gray')
plt.show()