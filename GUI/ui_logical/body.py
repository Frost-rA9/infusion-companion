import datetime
from functools import partial
from PySide2.QtGui import QPixmap
from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QGridLayout
import time
from PySide2.QtCore import QObject, Signal
from GUI.ui_logical.body_video import body_vedio
import cv2 as cv
from GUI.GUIHelper.QtImgConvert import QtImgConvert


# from DataProcess.Process import DataProcess


class BodySignal(QObject):
    sum_update = Signal(str)


class body:
    def __init__(self, ui_path, info):
        if ui_path:
            self.ui_body = QtHelper.read_ui(ui_path)
        else:
            # self.ui_body = QtHelper.read_ui("../ui/body.ui")
            self.ui_body = QtHelper.read_ui("../GUI/ui/body.ui")
        # self.data_process = DataProcess()
        self.info = info
        # 大视频的显示下标
        self.index = -1
        # 显示连接摄像头上限
        self.camera_sum = 20
        # 可用摄像头数
        self.camera_n = 0
        # 视频小组件列表
        self.control_list = []
        self.control_list_copy = []
        # 视频小组件的初始化
        self.g_layout = QGridLayout()
        self.control_init()
        # 时间差
        self.last_time = datetime.datetime.today().second
        # 摄像总数的事件实时监听
        self.body_signal = BodySignal()  # 实例化后才能用connect
        self.body_signal.sum_update.connect(self.ui_body.sum_line_edit.setText)
        self.ui_body.address_button.activated.connect(self.address_change)

    # 日志更新
    def info_update(self, info_m):
        self.info.add_info(info_m)

    # 位置变化
    def address_change(self):
        pass

    # 返回可用摄像头上限
    def return_camera_sum(self):
        return self.camera_sum

    # ui界面上可用摄像头个数发生改变
    def camera_update(self, camera_sum):
        if self.ui_body.sum_line_edit.text() != str(camera_sum):
            # print(f'原可用摄像头个数为{self.ui_body.sum_line_edit.text()}')
            self.body_signal.sum_update.emit(str(camera_sum))
            self.info_update(f'可用摄像头个数变化为{camera_sum}')

    # 选择摄像机的数字显示
    def number_update(self, i):
        self.ui_body.select_line_edit.setText(str(i))

    # 控件初始化
    def control_init(self):
        for i in range(self.camera_sum):
            b_v = body_vedio(None, str(i + 1))
            self.control_list.append(b_v)
            b_v.ui_body_video.number_button.clicked.connect(partial(self.number_update, i + 1))
            b_v.ui_body_video.number_button.clicked.connect(partial(self.video_show_change, i + 1))
            self.g_layout.addWidget(b_v.ui_body_video, i // 4, i % 4)
            self.g_layout.itemAt(i).widget().setHidden(True)
        self.ui_body.scrollAreaWidgetContents.setLayout(self.g_layout)

    # 隐藏视频小组件
    def control_hide(self, i):
        # self.g_layout.itemAt(i).widget().deleteLater()
        self.g_layout.itemAt(i).widget().setHidden(True)
        self.info_update(f'完成摄像头{i + 1}的隐藏')

    # 显示视频小组件
    def control_show(self, i):
        self.g_layout.itemAt(i).widget().setHidden(False)
        self.info_update(f'完成摄像头{i + 1}的显示')

    # 所有视频控件的图像显示及后端的启动
    def little_video_show(self, img, i: int):
        size = (50, 50)
        frame = cv.resize(img, size)
        convert_frame = QtImgConvert.CvImage_to_QImage(frame)
        self.g_layout.itemAt(i).widget().video.setPixmap(QPixmap.fromImage(convert_frame))
        now_time = datetime.datetime.today().second  # 获得当前时间
        if self.index == i + 1 and (now_time-self.last_time) >= 1:
            self.last_time = now_time
            # if self.index == i + 1:
            #     img, level, expression = self.data_process.process_seq(img)
            #     level, expression = str(level), str(expression)
            #     self.ui_body.liquid_level_line_edit.setText(level)
            #     self.ui_body.emotion_line_edit.setText(level)
            #     self.info_update("liquid level："+level)  # string
            #     self.info_update("expression："+expression)  # sring
            #     cv.imshow("loc img：", img)  # np.ndarray
            size = (640, 480)
            frame = cv.resize(img, size)
            convert_frame = QtImgConvert.CvImage_to_QImage(frame)
            self.ui_body.video.setPixmap(QPixmap.fromImage(convert_frame))
            # self.ui_body.video.clear()

    # 视频大组件的图像显示
    def video_show_change(self, i):
        self.index = i


if __name__ == '__main__':
    app = QApplication()
    b = body()
    b.ui_body.show()
    app.exec_()
