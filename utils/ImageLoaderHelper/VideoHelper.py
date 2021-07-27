# encoding=GBK
"""VideoHelper

    - 主要功能是在验证的时候起辅助作用
    - 提供静态的可迭代函数

    1. 从绝对路径的video中读取frame
    2. 从摄像头中读取frame
"""
import cv2 as cv
import numpy as np


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

    @staticmethod
    def read_frame_to_video(cap_number: int,
                            video_save_path: str,
                            video_time: int,
                            video_class: tuple = ('X', 'V', 'I', 'D')):
        """video_class
            cv2.VideoWriter_fourcc('I', '4', '2', '0'):该选项是一个未压缩的YUV颜色编码，是4:2:0色度子采样。这种编码兼容性很好，但文件较大，文件扩展名.avi。
            cv2.VideoWriter_fource('P', 'I', 'M', '1') ：该选项是MPEG-1编码类型，文件扩展名为.avi。
            cv2.VideoWriter_fource('X', 'V', 'I', 'D') ：给选项是MPEG-4编码类型，如果希望得到的视频大小为平均值，推荐这个选项，文件扩展名.avi。
            cv2.VideoWriter_fource('T', 'H', 'E', 'O') ： 该选项是Ogg Vorbis，文件扩展名应为.ogv。
            cv2.VideoWriter_fource('F', 'L', 'V', '1') ： 该选项是一个Flash视频，文件扩展名.flv。
        """
        cap = cv.VideoCapture(cap_number)
        fps = cap.get(cv.CAP_PROP_FPS)  # 看看摄像头的每秒帧率多少
        size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        c1, c2, c3, c4 = video_class
        video_writer = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc(c1, c2, c3, c4),
                                      fps, size)
        frame_number = video_time * fps - 1

        ret, frame = cap.read()
        while ret and frame_number > 0:
            cv.imshow("frame", frame)
            if not frame_number == np.inf:
                cv.waitKey(1)
            else:
                key = cv.waitKey(27)
                if key == ord('s'):
                    break
            video_writer.write(frame)
            ret, frame = cap.read()
            frame_number -= 1
        cap.release()


if __name__ == '__main__':
    img = "H:/DataSet/CAER/train/Anger/0001.avi"
    count = 0
    # for frame in VideoHelper.read_frame_from_video(img):
    #     count += 1
    # print(count)
    VideoHelper.read_frame_to_video(0, "test.avi", np.inf)






