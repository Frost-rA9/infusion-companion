# encoding=GBK
"""Cascade

        - 使用opencv中的CascadeClassifier进行分类
        - 由于权重位置固定，所以检测精度有限，但是速度较高

        - CascadeClassifier的参数
            1. image表示的是要检测的输入图像
            2. objects表示检测到的人脸目标序列
            3. scaleFactor表示每次图像尺寸减小的比例
            4. minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
            5. minSize为目标的最小尺寸
            6. minSize为目标的最大尺寸
            适当调整4,5,6两个参数可以用来排除检测结果中的干扰项。

"""
import numpy as np
import cv2 as cv


class Cascade:
    def __init__(self):
        self.faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, img: np.ndarray):
        pass
