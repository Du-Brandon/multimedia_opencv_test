import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

#復古CV效果風格轉換

def retor_cv_effect(img:np.ndarray) -> np.ndarray:
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
def watercolor(img: np.ndarray) -> np.ndarray:
    res = img.copy()

    #雙邊濾波
    for i in range(3):
        res = cv2.bilateralFilter(res, 9, 200, 200)

    #邊緣檢測
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    res = cv2.bitwise_and(res, edges)

    return res

# 定義油畫效果函數
def oil_paint(img: np.ndarray, radius: int, levels: int) -> np.ndarray:
    res = cv2.xphoto.oilPainting(img, radius, levels)
    return res

# 定義素描效果函數
def sketch_effect(img: np.ndarray) -> np.ndarray:
    # 轉換成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted_gray = 255 - gray

    blurred = cv2.GaussianBlur(inverted_gray, (51 , 51), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    #轉換色彩
    res = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    return res



img = None
img = cv2.imread("1.jpg")
# print(type(img))

if img is None:
    print("Image is not loaded")

retro = retor_cv_effect(img)
watercolor_img = watercolor(img)
oil_paint_img = oil_paint(img, 10, 1)
sketch_img = sketch_effect(img)

cv2.imwrite("retro.jpg", retro)
cv2.imwrite("watercolor.jpg", watercolor_img)
cv2.imwrite("oil_paint.jpg", oil_paint_img)
cv2.imwrite("sketch.jpg", sketch_img)