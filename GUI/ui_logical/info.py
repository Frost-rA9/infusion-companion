# encoding=GBK

from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import Signal, QObject

from threading import Thread


class InfoSignal(QObject):
    update_info_text_edit = Signal(str)
    clear_info_text_edit = Signal()


class info:
    def __init__(self,
                 ui_path: str = None):
        if ui_path:
            self.ui_info = QtHelper.read_ui(ui_path)
        else:
            # self.ui_info = QtHelper.read_ui("../ui/info.ui")
            self.ui_info = QtHelper.read_ui("../GUI/ui/info.ui")

        self.info_signal = InfoSignal()
        self.info_signal.update_info_text_edit.connect(self.ui_info.info_text_edit.append)  # 更新信号
        self.info_signal.clear_info_text_edit.connect(self.ui_info.info_text_edit.clear)  # 清空信号
        self.ui_info.info_text_edit.copyAvailable.connect(
            self.ui_info.info_text_edit.copy
        )  # 信息复制信号


if __name__ == '__main__':
    import time

    app = QApplication()
    i = info()
    i.ui_info.show()
    app.exec_()
