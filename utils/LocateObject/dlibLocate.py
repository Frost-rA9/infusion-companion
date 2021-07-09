# encoding=GBK
"""LocateRoI

    - 根据训练的向量机，返回起点和终点

    - 封装步骤：
        1. 转成灰度图
        2. 进行预测
        3. 返回预测区域的列表（注意，可能有多个）
            - 格式：[( (start_x, start_y), (end_x, end_y))..]

    - 关于参数
        1. 此类用于在内存中对ndarray的数据进行预测，所以不提供file形式的读取

"""
import numpy as np
import dlib
import cv2 as cv


class LocateRoI:
    def __init__(self, svm_path: str):
        self.detector = dlib.simple_object_detector(svm_path)

    def reload_detector(self, svm_path: str):
        self.detector = dlib.simple_object_detector(svm_path)

    def predict(self, img: np.ndarray):
        predict_list = []
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dist = self.detector(gray)
        for (k, d) in enumerate(dist):
            predict_list.append(((d.left(), d.top()), (d.left() + d.width(), d.top() + d.height())))
        return predict_list

    def predict_show(self,
                     img: np.ndarray,
                     color=(0, 255, 0),
                     think=1):
        predict_list = self.predict(img)
        for start, end in predict_list:
            cv.rectangle(img, start, end, color, think)
        # img = cv.resize(img, (1200, 800))
        cv.imshow("predict", img)
        cv.waitKey(0)
