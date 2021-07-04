# encoding=GBK
"""从ui文件中导入界面
    1. 使用QFile建立文件指针
    2. 建立QUiLoader对象并调用load方法
    3. 对ui文件中的事件进行设置
    4. ui.show()
"""
from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile


def print_hh():
    print("hh")

app = QApplication()
ui_status = QFile("./Status.ui")
ui = QUiLoader().load(ui_status)  # 返回内容实际是最顶层的窗体对象
ui.button.clicked.connect(print_hh)  # 所有组件变成ui的属性
ui.show()
app.exec_()