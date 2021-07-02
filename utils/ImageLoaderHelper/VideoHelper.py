# encoding=GBK
"""VideoHelper

    - 主要功能就是从视频中读取一帧帧的图像
    - 另外功能想到再加

    1. 提供一个可迭代的函数
        - 注意读取的时候要转化为绝对路径
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






