# -*- coding: utf-8 -*-

import cv2 #匯入OpenCV函式庫,用於影像處理
import mediapipe as mp #匯入MediaPipe函式庫,用於人臉檢測和特徵點追蹤
import numpy as np #匯入NumPy函式庫,用於數值運算
import time #匯入time模組,用於時間控制和延遲
# from google.colab.patches import cv2_imshow #用於在Colab環境中顯示OpenCV影像
# from IPython.display import display, Javascript, Image, HTML, clear_output #用於在Jupyter Notebook或Colab中顯示內容
# from google.colab.output import eval_js #用於在Colab中執行JavaScript程式碼 
from base64 import b64decode,b64encode #用於Base64編碼和解碼,處理影像資料傳輸 
import io # 用於處理二進位I/0
from PIL import Image as PILImage #匯入PIL的Image模組,用於影像處理


class SmileFilterApp:
    def _init_(self):

        # 初始化 MediaPipe Face Mesh 用於臉部特徵點檢測
        self.mp_face_mesh = mp.solutions. face_mesh
        self.face_mesh = self.mp_face_mesh. FaceMesh(
        max_num_faces=1, # 只檢測一張臉
        min_detection_confidence=0.5,#檢測信心閾值
        min_tracking_confidence=0.5)#追蹤信心閾值

        # 初始化 MediaPipe Drawing 用於繪製特徵點和連線
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)#設定繪製樣式

        # 設定濾鏡狀態相關變數
        self.filter_active = False #濾鏡是否啟用
        self.smile_counter=0 #計數檢測到微笑的連續幀數
        self.smile_threshold=3 #啟用濾鏡需要連續檢測到微笑的幀數閾值
        self.smile_detected_flag= False #是否曾經檢測到微笑的標記

        # 微笑最小值設定 - 濾鏡會在低於此值時自動取消
        self.min_smile_intensity=0.6 #微笑程度必須達到60%才會保持濾鏡效果

        #定義嗣鍵臉部特徵點索引
        #MediaPipe FaceMesh提供468個面部標記點,以下是嘴巴相關的重要標記點
        # 左嘴角的特徵點索引
        self.RIGHT_MOUTH_CORNER = 291 #右嘴角的特徵點索引
        # 上唇中間的特徵點索引
        # 下唇中間的特徵點索引

    def detect_smile(self, face_landmarks):
        # 計算嘴巴的寬度和高度
        left_mouth = face_landmarks. landmark[self.LEFT_MOUTH_CORNER]
        right_mouth = face_landmarks. landmark[self.RIGHT_MOUTH_CORNER]
        top_lip = face_landmarks.landmark[self.TOP_LIP]
        bottom_lip = face_landmarks. landmark[self.BOTTOM_LIP]

        # 計算嘴巴的寬度和高度
        mouth_width = abs(right_mouth.x- left_mouth.x) #嘴巴克度
        mouth_height = abs(top_lip.y - bottom_lip.y)

        # 計算嘴巴寬高比 - 微笑時嘴巴會變克而高度減少
        mouth_ratio=mouth_width/(mouth_height+1e-5)#加上極小值避免除以零

        # 計算嘴角的高度比例 - 檢查嘴角是否確實上揚
        left_corner_height = (left_mouth.y - top_lip.y)
        right_corner_height = (right_mouth.y - top_lip.y)
        corner_avg =(left_corner_height +right_corner_height)/ 2 #兩嘴角高度的平均值

        # 檢查中間嘴唇是否低於嘴角(真正的微笑特徴)
        middle_lip_y=(top_lip.y+bottom_lip.y)/2 #嘴唇中間點的垂直位置
        mouth_curve=(left_mouth.y+right_mouth.y)/2-middle_lip_y #嘴角與中間點的垂直差距

        # 修改判斷標準 - 更厳格的條件
        smile_threshold_ratio = 4.0
        smile_corner_threshold = 0.03

        # 要求嘴巴寬高比大於閾值、嘴角位置較高,以及嘴角確實上揚
        is_smiling = (mouth_ratio > smile_threshold_ratio and
        corner_avg < smile_corner_threshold and
        mouth_curve <- 0.01) #確保嘴巴呈微笑弧度

        # 計算微笑強度 - 用於調整濾鏡效果的強度
        # 將寬高比映射到0到1.5的強度值範圍
        smile_intensity =min((mouth_ratio- 3.5)/(smile_threshold_ratio- 3.5),1.5)#限制最大值

        return is_smiling, smile_intensity

        # 嘴巴高度

        # 左嘴角相封於上唇的位置
        # 右嘴角相封於上唇的位置

        # 嘴巴寬高比閾值
        # 嘴角高度閾值

def capture_and_display():
    # 初始化攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # 初始化 MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 只檢測一張臉
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    while True:
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 將影像轉換為 RGB 格式（MediaPipe 要求）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 將影像傳遞給 MediaPipe 進行處理
        results = face_mesh.process(rgb_frame)

        # 如果檢測到臉部，繪製特徵點
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # 顯示處理後的影像
        cv2.imshow('live', frame)

        # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('q'):
            break

    # 釋放攝影機裝置
    cap.release()
    cv2.destroyAllWindows()

capture_and_display()