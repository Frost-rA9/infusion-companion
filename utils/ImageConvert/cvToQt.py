import cv2
from PySide2 import QtGui


class cvToQt:
    @staticmethod
    def cvImgToQtImg(cvImg):
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGBA)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGBA8888)

        return QtImg