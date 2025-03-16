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
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)

    #歸一化
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    #二值化
    _, thresh = cv2.threshold(sobel, 128, 255, cv2.THRESH_BINARY)

    #輪廓檢測
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #合成圖片
    res = np.zeros(img.shape, dtype = np.uint8)

    # 繪製輪廓
    cv2.drawContours(res, contours, -1, (255, 255, 255), 1) #白色線條

    #添加網格
    h , w = img.shape[:2]
    for i in range(0, h, 10):
        cv2.line(res, (0, i), (w, i), (0, 50, 0), 1)
    for i in range(0, w, 10):
        cv2.line(res, (i, 0), (i, h), (0, 50, 0), 1)

    #添加雜訊
    noise = np.zeros(img.shape, np.uint8)
    cv2.randu(noise, 0, 50)
    res = cv2.add(res, noise)

    return res

#水彩畫風格轉換
def watercolor(img):
    res = img.copy()

    #雙邊濾波
    for i in range(3):
        res = cv2.bilateralFilter(res, 9, 75, 75)

    #邊緣檢測




img = None
img = cv2.imread("1.jpg")

if img is None:
    print("Image is not loaded")

retro = retor_cv_effect(img)

#用pillow顯示圖片
cv2.imwrite("retro.jpg", retro)