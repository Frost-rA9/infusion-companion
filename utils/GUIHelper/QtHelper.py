# encoding=GBK
"""QtHelper

    - Qt���������

    - ��Ҫʵ��һЩͨ�õľ�̬����
"""
import os
import PySide2
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader


class QtHelper:
    @staticmethod
    def add_to_system_path():
        dirname = os.path.dirname(PySide2.__file__)
        plugin_path = os.path.join(dirname, 'plugins', 'platforms')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path  # ���ȫ�ֱ���������
        print(plugin_path)

    @staticmethod
    def read_ui(ui_path: str):
        # UI�ļ���python����
        ui_file = QFile(ui_path)
        return QUiLoader().load(ui_file)


if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication
    # 1. ���Ի������������
    QtHelper.add_to_system_path()

    # 2. ����ui�ļ���ȡ
    app = QApplication()
    ui = QtHelper.read_ui("../../Test/Pyside2Usage/Complexui.ui")
    ui.show()
    app.exec_()
