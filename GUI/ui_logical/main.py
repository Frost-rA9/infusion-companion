from threading import Thread

from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QVBoxLayout, QWidget
from GUI.ui_logical.body import body
from GUI.ui_logical.head import head
from GUI.ui_logical.info import info


class main:
    def __init__(self,
                 ui_path: str = None):
        if ui_path:
            self.ui_main = QtHelper.read_ui(ui_path)
        else:
            # self.ui_main = QtHelper.read_ui("../ui/main.ui")
            self.ui_main = QtHelper.read_ui("../GUI/ui/main.ui")

        #     self.init_component()
        #
        # def init_component(self):
        self.widget = QWidget()
        self.v_layout = QVBoxLayout()

        self.i = info()
        self.b = body(None, self.i)
        self.h = head()
        self.h.ui_head.sum_label.setText(str(self.b.return_camera_sum()) + "个")

        self.v_layout.addWidget(self.h.ui_head)
        self.v_layout.addWidget(self.i.ui_info)
        self.v_layout.addWidget(self.b.ui_body)

        self.widget.setLayout(self.v_layout)
        self.ui_main.setCentralWidget(self.widget)
        # self.ui_main.setFixedSize(640, 1000)


if __name__ == '__main__':
    app = QApplication()
    m = main()
    m.ui_main.show()
    app.exec_()
