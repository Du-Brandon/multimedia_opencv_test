# -*- coding: utf-8 -*-

import cv2 #匯入OpenCV函式庫,用於影像處理
import mediapipe as mp #匯入MediaPipe函式庫,用於人臉檢測和特徵點追蹤
import numpy as np #匯入NumPy函式庫,用於數值運算
import time #匯入time模組,用於時間控制和延遲
import io # 用於處理二進位I/0
from base64 import b64decode,b64encode #用於Base64編碼和解碼,處理影像資料傳輸 
from PIL import Image as PILImage #匯入PIL的Image模組,用於影像處理

# 確認模組版本
def check_version():
    print("OpenCV version: ", cv2.__version__)
    print("MediaPipe version: ", mp.__version__)
    print("NumPy version: ", np.__version__)
    print("PIL version: ", PILImage.__version__)

check_version()