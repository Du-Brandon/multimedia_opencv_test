# -*- coding: utf-8 -*-

import cv2 #匯入OpenCV函式庫,用於影像處理
import mediapipe as mp #匯入MediaPipe函式庫,用於人臉檢測和特徵點追蹤
import numpy as np #匯入NumPy函式庫,用於數值運算
import time #匯入time模組,用於時間控制和延遲
# from google.colab.patches import cv2_imshow #用於在Colab環境中顯示OpenCV影像
# from IPython.display import display, Javascript, Image, HTML, clear_output #用於在Jupyter Notebook或Colab中顯示內容
# from google.colab.output import eval_js #用於在Colab中執行JavaScript程式碼 
import io # 用於處理二進位I/0
from base64 import b64decode,b64encode #用於Base64編碼和解碼,處理影像資料傳輸 
from PIL import Image as PILImage #匯入PIL的Image模組,用於影像處理

import os


class SmileFilterApp:
    def __init__(self):

        # 初始化 MediaPipe Face Mesh 用於臉部特徵點檢測
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
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
        self.LEFT_MOUTH_CORNER = 61
        self.RIGHT_MOUTH_CORNER = 291
        self.TOP_LIP = 13
        self.BOTTOM_LIP = 14

    # 檢測微笑的函數，回傳boolean值和微笑強度
    def detect_smile(self, face_landmarks) :
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

        # 修改判斷標準 - 更嚴格的條件
        smile_threshold_ratio = 2.5
        smile_corner_threshold = 0.05

        # 要求嘴巴寬高比大於閾值、嘴角位置較高,以及嘴角確實上揚
        is_smiling = (mouth_ratio > smile_threshold_ratio and corner_avg < smile_corner_threshold and mouth_curve <- 0.01) #確保嘴巴呈微笑弧度

        # 計算微笑強度 - 用於調整濾鏡效果的強度
        # 將寬高比映射到0到1.5的強度值範圍
        smile_intensity =min((mouth_ratio- 3.5)/(smile_threshold_ratio- 3.5),1.5)#限制最大值

        return is_smiling, smile_intensity
    
    # 濾鏡效果函數
    def black_and_white_filter(self, image):
        # 將影像轉換為灰階
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 將灰階影像轉換為三通道
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        return gray_image
    
    # 彩色濾鏡效果函數
    def color_filter(self, image):
        # 增強影像的色彩對比
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], 50)  # 增加飽和度
        hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], 50)  # 增加亮度
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return color_image

    # 素描濾鏡效果函數
    def sketch_filter(self, image):
        # 將影像轉換為灰階
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 反轉灰階影像
        inverted_image = cv2.bitwise_not(gray_image)
        # 使用高斯模糊
        blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
        # 將灰階影像與模糊影像結合，產生素描效果
        sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
        # 將單通道影像轉換為三通道
        sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2BGR)
        return sketch_image


def main():
    # 初始化 SmileFilterApp 物件
    smile_filter_app = SmileFilterApp()

    # 初始化攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 初始化濾鏡模式變數
    filter_mode = 'none'  # 預設無濾鏡

    while True:
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 將影像轉換為 RGB 格式（MediaPipe 要求）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 將影像傳遞給 MediaPipe 進行處理
        results = smile_filter_app.face_mesh.process(rgb_frame)

        # 如果檢測到臉部，進行微笑檢測並繪製特徵點
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 繪製臉部特徵點
                smile_filter_app.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=smile_filter_app.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=smile_filter_app.drawing_spec,
                    connection_drawing_spec=smile_filter_app.drawing_spec
                )

                # 執行微笑檢測
                is_smiling, smile_intensity = smile_filter_app.detect_smile(face_landmarks)

                # 如果檢測到微笑，顯示微笑狀態
                if is_smiling:
                            # 根據當前濾鏡模式應用濾鏡
                    if filter_mode == 'b':  # 灰階濾鏡
                        frame = smile_filter_app.black_and_white_filter(frame)
                    elif filter_mode == 'c':  # 彩色濾鏡
                        frame = smile_filter_app.color_filter(frame)
                    elif filter_mode == 's':  # 素描濾鏡
                        frame = smile_filter_app.sketch_filter(frame)
                    cv2.putText(frame, f"Smiling! Intensity: {smile_intensity:.2f}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Not Smiling",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



        # 顯示處理後的影像
        cv2.imshow('Smile Detection', frame)

        # 監聽按鍵輸入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按下 q 鍵退出
            break
        elif key == ord('b'):  # 按下 b 鍵切換到灰階濾鏡
            filter_mode = 'b'
        elif key == ord('c'):  # 按下 c 鍵切換到彩色濾鏡
            filter_mode = 'c'
        elif key == ord('s'):  # 按下 s 鍵切換到素描濾鏡
            filter_mode = 's'
        elif key == ord('n'):  # 按下 n 鍵取消濾鏡
            filter_mode = 'none'
        elif key == ord('p'):  # 按下 p 鍵擷取螢幕
            # 生成檔案名稱
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(output_dir, f"screenshot_{timestamp}.png")
            # 儲存影像
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

    # 釋放攝影機裝置
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()