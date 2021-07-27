# encoding=GBK
"""VideoHelper

    - ��Ҫ����������֤��ʱ����������
    - �ṩ��̬�Ŀɵ�������

    1. �Ӿ���·����video�ж�ȡframe
    2. ������ͷ�ж�ȡframe
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
            cv2.VideoWriter_fourcc('I', '4', '2', '0'):��ѡ����һ��δѹ����YUV��ɫ���룬��4:2:0ɫ���Ӳ��������ֱ�������Ժܺã����ļ��ϴ��ļ���չ��.avi��
            cv2.VideoWriter_fource('P', 'I', 'M', '1') ����ѡ����MPEG-1�������ͣ��ļ���չ��Ϊ.avi��
            cv2.VideoWriter_fource('X', 'V', 'I', 'D') ����ѡ����MPEG-4�������ͣ����ϣ���õ�����Ƶ��СΪƽ��ֵ���Ƽ����ѡ��ļ���չ��.avi��
            cv2.VideoWriter_fource('T', 'H', 'E', 'O') �� ��ѡ����Ogg Vorbis���ļ���չ��ӦΪ.ogv��
            cv2.VideoWriter_fource('F', 'L', 'V', '1') �� ��ѡ����һ��Flash��Ƶ���ļ���չ��.flv��
        """
        cap = cv.VideoCapture(cap_number)
        fps = cap.get(cv.CAP_PROP_FPS)  # ��������ͷ��ÿ��֡�ʶ���
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






