# encoding=GBK
"""��ui�ļ��е������
    1. ʹ��QFile�����ļ�ָ��
    2. ����QUiLoader���󲢵���load����
    3. ��ui�ļ��е��¼���������
    4. ui.show()
"""
from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile


def print_hh():
    print("hh")

app = QApplication()
ui_status = QFile("./Status.ui")
ui = QUiLoader().load(ui_status)  # ��������ʵ�������Ĵ������
ui.button.clicked.connect(print_hh)  # ����������ui������
ui.show()
app.exec_()