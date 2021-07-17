# encoding=GBK
"""Cascade

        - 使用opencv中的CascadeClassifier进行分类
        - 由于权重位置固定，所以检测精度有限，但是速度较高
        - 采用更激进的方案，只保留一个

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
        faces, reject_levels, level_weights = self.faceCascade.detectMultiScale3(img, 1.1, 4, outputRejectLevels=True)
        l = []
        if len(reject_levels) == 0:
            return l  # 防止后续的解码

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
        """直接返回绘制好的图像"""
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


