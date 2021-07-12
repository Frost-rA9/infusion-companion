# encoding=GBK
"""Cascade

        - ʹ��opencv�е�CascadeClassifier���з���
        - ����Ȩ��λ�ù̶������Լ�⾫�����ޣ������ٶȽϸ�

        - CascadeClassifier�Ĳ���
            1. image��ʾ����Ҫ��������ͼ��
            2. objects��ʾ��⵽������Ŀ������
            3. scaleFactor��ʾÿ��ͼ��ߴ��С�ı���
            4. minNeighbors��ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽����),
            5. minSizeΪĿ�����С�ߴ�
            6. minSizeΪĿ������ߴ�
            �ʵ�����4,5,6�����������������ų�������еĸ����

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
        """ֱ�ӷ��ػ��ƺõ�ͼ��"""
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