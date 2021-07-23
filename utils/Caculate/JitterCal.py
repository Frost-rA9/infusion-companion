# encoding=GBK
"""JitterCal

    - 图形抖动计算

    - 使用比较简洁的方案进行检测，防止计算量过大导致的帧率进一步下降
"""

import cv2 as cv
import numpy as np


class JitterCal:
    print_var = False

    @staticmethod
    def jitter_detect(pre_img: np.ndarray, next_img: np.ndarray):
        pre_mean = JitterCal.cal_gray_img_mean(pre_img)
        next_mean = JitterCal.cal_gray_img_mean(next_img)

        differ = abs(next_mean - pre_mean) / pre_mean
        if JitterCal.print_var:
            print("Jitter cal differ is:", differ)

        if differ > 0.01:
            return True
        else:
            return False

    @staticmethod
    def cal_gray_img_mean(img: np.ndarray):
        if len(img.shape) != 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mean = 0
        for i in range(img.shape[0]):
            mean += np.mean(img[i])
        return mean