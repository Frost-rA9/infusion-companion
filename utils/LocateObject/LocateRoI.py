# encoding=GBK
"""LocateRoI

    - ����ѵ���������������������յ�

    - ��װ���裺
        1. ת�ɻҶ�ͼ
        2. ����Ԥ��
        3. ����Ԥ��������б�ע�⣬�����ж����
            - ��ʽ��[( (start_x, start_y), (end_x, end_y))..]

    - ���ڲ���
        1. �����������ڴ��ж�ndarray�����ݽ���Ԥ�⣬���Բ��ṩfile��ʽ�Ķ�ȡ

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
