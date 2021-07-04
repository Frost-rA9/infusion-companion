import cv2
from PySide2 import QtGui


class cvToQt:
    @staticmethod
    def cvImgToQtImg(cvImg):
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)

        return QtImg