# encoding=GBK
from GUI.GUIHelper.QtImgConvert import QtImgConvert
from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QPixmap
import cv2 as cv
import numpy as np
from threading import Thread


class video:
    def __init__(self,
                 ui_path: str = None):
        if ui_path:
            self.ui_video = QtHelper.read_ui(ui_path)
        else:
            self.ui_video = QtHelper.read_ui("../ui/video.ui")


'''
        self.main_thread = Thread(target=self.main_loop)
        self.main_thread.start()

    def main_loop(self):
        # 用于测试图像与视频流的显示
        # img = cv.imread("../../Resource/CAER-S/Test/Anger/0001.png")
        # convert_img = QtImgConvert.CvImage_to_QImage(img)
        vid = "../../Resource/CAER/TEST/Anger/0001.avi"
        cap = cv.VideoCapture(vid)
        ret, frame = cap.read()
        while ret:
            frame = cv.resize(frame, (640, 480))
            convert_frame = QtImgConvert.CvImage_to_QImage(frame)
            self.ui_video.video.setScaledContents(True)
            self.ui_video.video.setPixmap(QPixmap.fromImage(convert_frame))
            ret, frame = cap.read()
            self.ui_video.show()
            cv.waitKey(30)
        cap.release()
        # v.ui_video.video.setScaledContents(True)
        # v.ui_video.video.setPixmap(QPixmap.fromImage(convert))
        # v.ui_video.show()
'''

if __name__ == '__main__':
    app = QApplication()
    v = video()
    v.ui_video.show()
    app.exec_()
    '''
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
'''
