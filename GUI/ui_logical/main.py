from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QVBoxLayout


class main:
    def __init__(self,
                 parent=None,
                 ui_path: str = None):
        if ui_path:
            self.ui_main = QtHelper.read_ui(ui_path)
        else:
            self.ui_main = QtHelper.read_ui("../ui/main.ui")

