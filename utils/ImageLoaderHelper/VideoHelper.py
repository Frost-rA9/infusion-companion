# encoding=GBK
"""VideoHelper

    - ��Ҫ���ܾ��Ǵ���Ƶ�ж�ȡһ֡֡��ͼ��
    - ���⹦���뵽�ټ�

    1. �ṩһ���ɵ����ĺ���
        - ע���ȡ��ʱ��Ҫת��Ϊ����·��
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


if __name__ == '__main__':
    img = "H:/DataSet/CAER/train/Anger/0001.avi"
    count = 0
    for frame in VideoHelper.read_frame_from_video(img):
        count += 1
    print(count)






