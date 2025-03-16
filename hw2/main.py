import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

#復古CV效果風格轉換

def retor_cv_effect(img):
    #轉換成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #使用Sobel算子檢測邊緣
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)








img = None
img = cv2.imread("1.jpg")

if img is None:
    print("Image is not loaded")

