# encoding=GBK
"""VideoHelper

    - ��Ҫ����������֤��ʱ����������
    - �ṩ��̬�Ŀɵ�������

    1. �Ӿ���·����video�ж�ȡframe
    2. ������ͷ�ж�ȡframe
"""
import cv2 as cv


class VideoHelper:
    @staticmethod
    def read_frame_from_video(video_path: str):
        video = cv.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield frame
            else:
                break

    @staticmethod
    def read_frame_from_cap(cap_number: int):
        cap = cv.VideoCapture(cap_number)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break


if __name__ == '__main__':
    img = "H:/DataSet/CAER/train/Anger/0001.avi"
    count = 0
    for frame in VideoHelper.read_frame_from_video(img):
        count += 1
    print(count)






