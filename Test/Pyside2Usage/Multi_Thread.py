"""多线程后台任务
    1. QThread: 使用Qt专有的event才用其他相同
    2. Thread

    - 子线程操作界面可能会导致主界面显示崩溃
    - 所以子线程主要发送信号，主线程轮询接收信号，由一条主线程进行更新

    子线程发出信号的步骤：
        1. 继承QObject
        2. 主线程通过connect方法处理signal信号
        3. 子线程操作界面的时候发出emit信号，里面包含必要的桉树
        4. 住线程信号处理函数，执行处理
"""
import time

from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QTextBrowser


class MySignals(QObject):
    # 定义一种信号，参数分别为QTextBrowser和字符串
    text_print = Signal(QTextBrowser, str)   # 1. 声明信号
    # 还可以定义其他信号
    update_table = Signal(str)


class Stats:
    def __init__(self):
        self.ms = MySignals()
        self.ms.text_print.connect(self.print_to_gui)  # 2. 绑定事件处理函数

    def print_to_gui(self, fb, text):
        fb.append(str(text))

    def click(self):
        for i in range(6):
            self.ms.text_print.emit("hh")  # 3. 发出信号

