from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication


class head:
    def __init__(self, ui_path: str = None):
        if ui_path:
            self.ui_head = QtHelper.read_ui(ui_path)
        else:
            self.ui_head = QtHelper.read_ui("../ui/head.ui")

        print(self.ui_head)


if __name__ == '__main__':
    pass
    app = QApplication()
    h = head()
    h.ui_head.show()
    app.exec_()