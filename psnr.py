# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:22:52 2020

@author: harshvardhan
"""

from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  

original = cv2.imread("test.jpg") 
compressed = cv2.imread("shapes-lap.jpg") 
value = PSNR(original, compressed) 
print(f"PSNR value is {value} dB") 
