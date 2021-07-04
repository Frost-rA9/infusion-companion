"""���̺߳�̨����
    1. QThread: ʹ��Qtר�е�event����������ͬ
    2. Thread

    - ���̲߳���������ܻᵼ����������ʾ����
    - �������߳���Ҫ�����źţ����߳���ѯ�����źţ���һ�����߳̽��и���

    ���̷߳����źŵĲ��裺
        1. �̳�QObject
        2. ���߳�ͨ��connect��������signal�ź�
        3. ���̲߳��������ʱ�򷢳�emit�źţ����������Ҫ������
        4. ס�߳��źŴ�������ִ�д���
"""
import time

from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QTextBrowser


class MySignals(QObject):
    # ����һ���źţ������ֱ�ΪQTextBrowser���ַ���
    text_print = Signal(QTextBrowser, str)   # 1. �����ź�
    # �����Զ��������ź�
    update_table = Signal(str)


class Stats:
    def __init__(self):
        self.ms = MySignals()
        self.ms.text_print.connect(self.print_to_gui)  # 2. ���¼�������

    def print_to_gui(self, fb, text):
        fb.append(str(text))

    def click(self):
        for i in range(6):
            self.ms.text_print.emit("hh")  # 3. �����ź�

