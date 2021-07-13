from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QPushButton
from threading import Thread
import cv2 as cv


class body_vedio:
    def __init__(self, ui_path, object_name):
        if ui_path:
            self.ui_body_video = QtHelper.read_ui(ui_path)
        else:
            self.ui_body_video = QtHelper.read_ui("../ui/body_video.ui")
        self.ui_body_video.number_button.setText("摄像" + object_name)
    #     self.main_thread = Thread(target=self.main_loop)
    #     self.main_thread.start()
    #
    # def main_loop(self):
    #     vid = "../../Resource/CAER/TEST/Anger/0001.avi"
    #     cap = cv.VideoCapture(0)
    #     ret, frame = cap.read()
    #     while ret:
    #         # frame = cv.resize(frame, (640, 480))
    #         convert_frame = QtImgConvert.CvImage_to_QImage(frame)
    #         self.ui_video.video.setScaledContents(True)
    #         self.ui_video.video.setPixmap(QPixmap.fromImage(convert_frame))
    #         ret, frame = cap.read()
    #         self.ui_video.show()
    #         cv.waitKey(30)
    #     cap.release()


if __name__ == '__main__':
    app = QApplication()
    bv = body_vedio()
    bv.ui_body_video.show()
    app.exec_()
