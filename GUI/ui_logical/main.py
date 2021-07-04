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
            self.ui_main = QtHelper.read_ui("../ui/main.ui")

        self.init_component()

    def init_component(self):
        widget = QWidget()
        v_layout = QVBoxLayout()

        b = body()
        h = head()
        i = info()

        v_layout.addWidget(h.ui_head)
        v_layout.addWidget(b.ui_body)
        v_layout.addWidget(i.ui_info)

        widget.setLayout(v_layout)
        self.ui_main.setCentralWidget(widget)


if __name__ == '__main__':
    app = QApplication()
    m = main()
    m.ui_main.show()
    app.exec_()





