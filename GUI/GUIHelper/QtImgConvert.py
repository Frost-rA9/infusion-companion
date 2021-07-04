# encoding=GBK
"""QtImgConvert

    - 主要作用是完成opencv的numpy.ndarray到QImage
    - 还有反方向的相互转换

    - 提供静态函数
"""

import numpy as np
from PySide2.QtGui import QImage
from PySide2.QtGui import QPixmap
import cv2 as cv


class QtImgConvert:
    @staticmethod
    def CvImage_to_QImage(array_img: np.ndarray):
        height, width, depth = array_img.shape
        q_img = QImage(array_img.data, width, height, width * depth, QImage.Format_RGB888).rgbSwapped()
        return q_img

    @staticmethod
    def QPixmap_to_CvImage(array_img: QPixmap):
        q_img = array_img.toImage()
        temp_shape = (q_img.height(), q_img.bytesPerLine() * 8 // q_img.depth())
        temp_shape += (4,)
        ptr = q_img.bits()
        ptr.setSize(q_img.byteCount())
        cv_img = np.array(ptr, dtype=np.uint8, ).reshape(temp_shape)
        cv_img = cv_img[..., 3]
        return cv_img
