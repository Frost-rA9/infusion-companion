from functools import partial
from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide2.QtWidgets import QPushButton
from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QLabel, QGridLayout
from PySide2.QtCore import QObject, Signal
from threading import Thread
import threading
from GUI.ui_logical.body_video import body_vedio
import time
from GUI.camera.number import number
from GUI.camera.video_image import video_image
import inspect
import ctypes


class BodySignal(QObject):
    count_update = Signal(str)


class body:
    def __init__(self, ui_path: str = None):
        if ui_path:
            self.ui_body = QtHelper.read_ui(ui_path)
        else:
            self.ui_body = QtHelper.read_ui("../ui/body.ui")
        self.camera_n = number().index
        self.g_layout = QGridLayout()
        self.t = video_image(self.ui_body, -1, True)
        self.add()
        self.body_signal = BodySignal()  # 实例化后才能用connect
        self.body_signal.count_update.connect(self.ui_body.sum_line_edit.setText)

        self.main_thread = Thread(target=self.main_loop)
        self.main_thread.start()

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    def main_loop(self):
        while True:
            self.camera_n = number().index  # 运行的摄像机个数
            if self.ui_body.sum_line_edit.text() != self.camera_n:
                self.body_signal.count_update.emit(str(self.camera_n))
            time.sleep(10)

    # 选择摄像机的数字显示
    def number_update(self, i):
        self.ui_body.select_line_edit.setText(str(i))

    def choose_video(self, i):
        # self.stop_thread(self.t.main_thread)
        # self.t = video_image(self.ui_body, i, True)
        self.t.camera_number = i

    # 小视频组件添加
    def add(self):
        for i in range(self.camera_n):
            # for i in range(1):
            b_v = body_vedio(None, str(i + 1))
            b_v.ui_body_video.number_button.clicked.connect(partial(self.choose_video, i))
            b_v.ui_body_video.number_button.clicked.connect(partial(self.number_update, i + 1))
            # b_v.ui_body_video.number_button.clicked.connect(self.choose_video)
            video_image(b_v.ui_body_video, i, False)
            self.g_layout.addWidget(b_v.ui_body_video, i // 4, i % 4)
        self.ui_body.scrollAreaWidgetContents.setLayout(self.g_layout)


if __name__ == '__main__':
    app = QApplication()
    b = body()
    b.ui_body.show()
    app.exec_()
