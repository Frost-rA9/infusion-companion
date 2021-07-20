# encoding=GBK
"""head

    - �������֮һ
    - �˺��������Ҫ��������
        1. ϰ��SLOT,SIGNAL,emit�ĸ���
        2. ��ϰ���߳�
        3. ����������룬��װ�ص�������Ĳ���
            - ����setCentralWidget��������һ������
            - ���Խ�ȫ�����ͨ��initװ�ص�һ�������в�����ΪCenterlWidget

    - ������һ���̣߳�ÿ��һ��ʱ���������ݽ�����ѯ
    1. ÿ��1s�����źŸ���datetime
    2. ͬ��ÿ1sѯ��TCP�����ӱ���,����б仯�����������ź�
        - ��TCPд������
"""

from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QLabel
from PySide2.QtCore import QObject, Signal
from threading import Thread
import time
import datetime


class HeadSignal(QObject):
    # ����head��ȫ����signal
    date_update = Signal(str)


class head:
    def __init__(self,
                 ui_path: str = None):
        # 1. ����ui�ļ�
        if ui_path:
            self.ui_head = QtHelper.read_ui(ui_path)
        else:
            # self.ui_head = QtHelper.read_ui("../ui/head.ui")
            self.ui_head = QtHelper.read_ui("../GUI/ui/head.ui")

        # m.ui_main.setCentralWidget(self.ui_head)

        # 2. ��ʼ���ź����࣬�����ź���۵�����
        self.head_signal = HeadSignal()  # ʵ�����������connect
        self.head_signal.date_update.connect(self.ui_head.datetime.setText)

        # 3. ������Ҫ���߳�
        self.main_thread = Thread(target=self.main_loop)
        self.main_thread.start()

    def main_loop(self):
        while True:
            time.sleep(1)
            now = str(datetime.datetime.now())[:-7]
            self.head_signal.date_update.emit(now)
            # print(self.ui_head.datetime.text())


if __name__ == '__main__':

    app = QApplication()
    h = head()
    h.ui_head.show()
    # m.ui_main.show()
    app.exec_()
