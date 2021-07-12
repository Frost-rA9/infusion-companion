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
        pass
