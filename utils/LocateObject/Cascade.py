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
        faces = self.faceCascade.detectMultiScale(img, 1.1, 4)
        l = []
        for left, top, width, height in faces:
            l.append(( (left, top), (left + width, top + height) ))
        return l

    def plot_rect(self, img):
        """直接返回绘制好的图像"""
        l = self.detect_face(img)
        for (left, top), (right, end) in l:
            cv.rectangle(img, (left, top), (right, end), color=(0, 0, 255), thickness=2)
        return img


if __name__ == '__main__':
    img_path = "../../Resource/face_test.png"
    img = cv.imread(img_path)
    c = Cascade()
    img = c.plot_rect(img)
    cv.imshow("face_rect", img)
    cv.waitKey(0)