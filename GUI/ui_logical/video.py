from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication, QVBoxLayout


class video:
    def __init__(self,
                 parent=None,
                 ui_path: str = None):
        if ui_path:
            self.ui_video = QtHelper.read_ui(ui_path)
        else:
            self.ui_video = QtHelper.read_ui("../ui/video.ui")