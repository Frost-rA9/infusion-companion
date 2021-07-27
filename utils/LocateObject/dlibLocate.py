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
    print_var = False  # 用来控制是否打印中间信息

    def __init__(self, svm_path: str, rectangle_area: float = None):
        self.detector = dlib.simple_object_detector(svm_path)
        self.rectangle_area = rectangle_area  # 用来过滤面积过小的区域

    def reload_detector(self, svm_path: str):
        self.detector = dlib.simple_object_detector(svm_path)

    def predict(self, img: np.ndarray):
        predict_list = []
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dist = self.detector(gray)
        for (k, d) in enumerate(dist):
            left, top, right, bottom = d.left(), d.top(), d.left() + d.width(), d.top() + d.height()

            # 过滤面积太小的区域
            if self.rectangle_area:
                if abs(right - left) * abs(bottom - right) < self.rectangle_area:
                    pass

            predict_list.append(((left, top), (right, bottom)))
        return predict_list

    def predict_show(self,
                     img: np.ndarray,
                     color=(0, 255, 0),
                     think=1):
        predict_list = self.predict(img)
        for start, end in predict_list:
            cv.rectangle(img, start, end, color, think)
        return img


if __name__ == '__main__':
    locate = LocateRoI("../../Resource/svm/trained/new_bottle_svm.svm")
    # from utils.ImageLoaderHelper.VideoHelper import VideoHelper
    # for frame in VideoHelper.read_frame_from_cap(0):
    #     img = locate.predict_show(frame)
    #     cv.imshow("img", img)
    #     cv.waitKey(1)
    import os
    import cv2 as cv
    path = "H:/bottle"
    for d in os.listdir(path):
        file = path + "/" + d
        img = cv.imread(file)
        img = locate.predict_show(img)
        cv.imshow("img", img)
        cv.waitKey(100)


