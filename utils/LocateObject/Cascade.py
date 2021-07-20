# encoding=GBK
"""Cascade

        - ʹ��opencv�е�CascadeClassifier���з���
        - ����Ȩ��λ�ù̶������Լ�⾫�����ޣ������ٶȽϸ�
        - ���ø������ķ�����ֻ����һ��

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
        faces, reject_levels, level_weights = self.faceCascade.detectMultiScale3(img, 1.1, 4, outputRejectLevels=True)
        l = []
        if len(reject_levels) == 0:
            return l  # ��ֹ�����Ľ���

        max_index = list(reject_levels).index(max(reject_levels))
        print("reject_level", reject_levels[max_index], "level_wight", level_weights[max_index])
        if level_weights[max_index] > 4:
            left, top, width, height = faces[max_index]
            l.append(((left, top), (left + width, top + height)))
        return l
        # for index in range(len(faces)):
        #     rej_level = reject_levels[index]
        #     if rej_level < 4:
        #         pass
        #     else:
        #         print("reject_level", reject_levels[index], "level_wight", level_weights[index])
        #     left, top, width, height = faces[index]
        #     l.append(((left, top), (left + width, top + height)))
        # return l

    def plot_rect(self, img):
        """ֱ�ӷ��ػ��ƺõ�ͼ��"""
        l = self.detect_face(img)
        for (left, top), (right, end) in l:
            cv.rectangle(img, (left, top), (right, end), color=(0, 0, 255), thickness=2)
        return img


if __name__ == '__main__':
    from utils.ImageLoaderHelper.VideoHelper import VideoHelper
    c = Cascade()

    for frame in VideoHelper.read_frame_from_cap(0):
        img = c.plot_rect(frame)
        cv.imshow("img", img)
        cv.waitKey(1)


