# encoding=GBK
from GUI.GUIHelper.QtImgConvert import QtImgConvert
from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QPixmap
import cv2 as cv
import numpy as np


class VideoSignal(QObject):
    update_video = Signal(np.ndarray)


class video:
    def __init__(self,
                 ui_path: str = None):
        if ui_path:
            self.ui_video = QtHelper.read_ui(ui_path)
        else:
            self.ui_video = QtHelper.read_ui("../ui/video.ui")


if __name__ == '__main__':
    # 用于测试图像与视频流的显示
    img = cv.imread("../../Resource/CAER-S/Test/Anger/0001.png")
    convert_img = QtImgConvert.CvImage_to_QImage(img)
    vid = "../../Resource/CAER/TEST/Anger/0001.avi"
    cap = cv.VideoCapture(vid)
    app = QApplication()
    v = video()
    ret, frame = cap.read()
    while ret:
        convert_frame = QtImgConvert.CvImage_to_QImage(frame)
        v.ui_video.video.setScaledContents(True)
        v.ui_video.video.setPixmap(QPixmap.fromImage(convert_frame))
        ret, frame = cap.read()
        v.ui_video.show()
        cv.waitKey(30)
    cap.release()
    # v.ui_video.video.setScaledContents(True)
    # v.ui_video.video.setPixmap(QPixmap.fromImage(convert))
    # v.ui_video.show()
    app.exec_()
