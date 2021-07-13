from threading import Thread
import cv2 as cv
import time

from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QLabel

from GUI.GUIHelper.QtImgConvert import QtImgConvert
import threading
import inspect
import ctypes


class video_image:
    # flag=false为正常图像
    def __init__(self, control: QLabel, camera_number: int, flag: bool):
        self.control = control
        self.camera_number = camera_number
        self.flag = flag
        self.size = (50, 50)
        self.jud()
        self.main_thread = Thread(target=self.main_loop)
        self.main_thread.start()

    # 判断尺寸
    def jud(self):
        if self.flag:
            self.size = (640, 480)

    def main_loop(self):
        while self.camera_number == -1:
            time.sleep(1)
            pass
        last_number = self.camera_number
        cap = cv.VideoCapture(self.camera_number, cv.CAP_DSHOW)
        # cap = cv.VideoCapture(self.camera_number)
        ret, frame = cap.read()
        while ret:
            frame = cv.resize(frame, self.size)
            convert_frame = QtImgConvert.CvImage_to_QImage(frame)
            # self.control.setScaledContents(True)
            self.control.video.setPixmap(QPixmap.fromImage(convert_frame))
            ret, frame = cap.read()
            self.control.show()
            cv.waitKey(30)
            if last_number != self.camera_number:
                cap.release()
                last_number = self.camera_number
                cap = cv.VideoCapture(self.camera_number, cv.CAP_DSHOW)
                # cap = cv.VideoCapture(self.camera_number)
                ret, frame = cap.read()
                print(1)
        cap.release()
