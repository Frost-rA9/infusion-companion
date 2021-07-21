# encoding=GBK
from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication
import time


# class InfoSignal(QObject):
#     update_info_text_edit = Signal(str)
#     clear_info_text_edit = Signal()


class info:
    def __init__(self,
                 ui_path: str = None):
        if ui_path:
            self.ui_info = QtHelper.read_ui(ui_path)
        else:
            # self.ui_info = QtHelper.read_ui("../ui/info.ui")
            self.ui_info = QtHelper.read_ui("../GUI/ui/info.ui")

    def add_info(self, info):
        self.ui_info.info_text_edit.append(str(time.ctime()) + '��' + info)

    # self.info_signal = InfoSignal()
    # self.info_signal.update_info_text_edit.connect(self.ui_info.info_text_edit.append)  # �����ź�
    # self.info_signal.clear_info_text_edit.connect(self.ui_info.info_text_edit.clear)  # ����ź�
    # self.ui_info.info_text_edit.copyAvailable.connect(self.ui_info.info_text_edit.copy)  # ��Ϣ�����ź�


if __name__ == '__main__':
    import time

    app = QApplication()
    i = info()
    i.ui_info.show()
    app.exec_()
