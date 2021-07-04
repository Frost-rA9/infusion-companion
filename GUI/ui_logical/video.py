# encoding=GBK

from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QImage
from PySide2.QtGui import QPixmap
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
    img = "../../Resource/CAER-S/Test/Anger/0001.png"
    import cv2 as cv
    from GUI.GUIHelper.QtImgConvert import QtImgConvert
    i = cv.imread(img)
    convert = QtImgConvert.ndarray_to_QImage(i)
    app = QApplication()
    v = video()
    pix = QPixmap("../../Resource/CAER-S/Test/Anger/0001.png")
    v.ui_video.video.setScaledContents(True)
    v.ui_video.video.setPixmap(QPixmap.fromImage(convert))
    v.ui_video.show()
    app.exec_()
