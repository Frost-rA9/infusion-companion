# encoding=GBK
"""head

    - 核心组件之一
    - 此核心组件主要用来试手
        1. 习惯SLOT,SIGNAL,emit的概念
        2. 练习多线程
        3. 关于组件分离，并装载到主界面的步骤
            - 调用setCentralWidget可以设置一个部件
            - 可以将全部组件通过init装载到一个布局中并设置为CenterlWidget

    - 本身开启一个线程，每隔一定时间对相关内容进行轮询
    1. 每隔1s发出信号更新datetime
    2. 同样每1s询问TCP的连接变量,如果有变化，发出更新信号
        - 在TCP写完后添加
"""

from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QLabel
from PySide2.QtCore import QObject, Signal
from threading import Thread
import time
import datetime


class HeadSignal(QObject):
    # 定义head的全部的signal
    date_update = Signal(str)
    connect_number_update = Signal(str)


class head:
    def __init__(self, ui_path: str = None):
        # 1. 导入ui文件
        if ui_path:
            self.ui_head = QtHelper.read_ui(ui_path)
        else:
            self.ui_head = QtHelper.read_ui("../ui/head.ui")

        # m.ui_main.setCentralWidget(self.ui_head)

        # 2. 初始化信号量类，并且信号与槽的连接
        self.head_signal = HeadSignal()  # 实例化后才能用connect
        self.head_signal.date_update.connect(self.update_datetime)

        # 3. 启动主要的线程
        self.main_thread = Thread(target=self.main_loop)
        self.main_thread.start()

    def main_loop(self):
        while True:
            time.sleep(1)
            now = str(datetime.datetime.now())[:-7]
            self.head_signal.date_update.emit(now)
            # print(self.ui_head.datetime.text())

    """SLOT函数"""

    def update_datetime(self, date):
        self.ui_head.datetime.setText(date)

    def update_connect_number(self, number):
        self.ui_head.connect_number.setText(number)


if __name__ == '__main__':
    from GUI.ui_logical.main import main

    app = QApplication()
    m = main()
    h = head()
    h.ui_head.show()
    # m.ui_main.show()
    app.exec_()
