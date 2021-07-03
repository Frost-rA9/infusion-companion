from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication

app = QApplication()
# 1. 添加主窗口图像
app.setWindowIcon(QIcon("logo.png"))

# 2. 应用程序图标.exe的
# 编译时添加命令--icon="logo.ico"
# 只能添加.ico文件

