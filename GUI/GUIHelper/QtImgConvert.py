# encoding=GBK
"""QtImgConvert

    - 主要作用是完成opencv的numpy.ndarray到QImage
    - 还有反方向的相互转换

    - 提供静态函数
"""

import numpy as np
from PySide2.QtGui import QImage
import cv2 as cv


class QtImgConvert:
    @staticmethod
    def ndarray_to_QImage(array_img: np.ndarray):
        height, width, depth = array_img.shape
        # array_img = cv.cvtColor(array_img, cv.COLOR_BGR2RGB)
        q_img = QImage(array_img.data, width, height, width * depth, QImage.Format_RGB888).rgbSwapped()
        return q_img

    @staticmethod
    def QImage_to_ndarray():
        pass
