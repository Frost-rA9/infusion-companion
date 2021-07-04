from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QVBoxLayout


class body:
    def __init__(self,
                 parent=None,
                 ui_path: str = None):
        if ui_path:
            self.ui_body = QtHelper.read_ui(ui_path)
        else:
            self.ui_body = QtHelper.read_ui("../ui/body.ui")