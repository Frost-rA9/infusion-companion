# encoding=GBK
"""示例程序：薪资统计

    - 控件
        1. 控件创建
        2. 控件信息处理
        3. 事件连接
"""
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox


def handleCalc():
    info = text_edit.toPlainText()
    QMessageBox.about(window, '统计结果', info)  # 弹出通知对话框


app = QApplication([])  # 启动图形界面的底层功能，初始化各种信息等，派发信息等

window = QMainWindow()
window.resize(500, 400)
window.move(300, 310)  # 出现在屏幕的位置, 横向300，纵向310
window.setWindowTitle("薪资统计")

text_edit = QPlainTextEdit(window)  # 这个窗口的父窗口
text_edit.setPlaceholderText("请输入信息表")  # 提示信息，没有信息时显示
text_edit.move(10, 25)  # 相对父窗口的位置
text_edit.resize(300, 350)

button = QPushButton('统计', window)
button.move(330, 120)
button.clicked.connect(handleCalc)  # 把signal(clicked)连接到SLOT(handleCalc)从而绑定操作

window.show()  # 执行最上层的show可以显示全部内容
app.exec_()  # 进入事件处理循环
