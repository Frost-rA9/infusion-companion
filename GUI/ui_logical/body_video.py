from GUI.GUIHelper.QtHelper import QtHelper
from PySide2.QtWidgets import QApplication


# 小视频label
class body_vedio:
    def __init__(self, ui_path, object_name):
        if ui_path:
            self.ui_body_video = QtHelper.read_ui(ui_path)
        else:
            # self.ui_body_video = QtHelper.read_ui("../ui/body_video.ui")
            self.ui_body_video = QtHelper.read_ui("../GUI/ui/body_video.ui")
        self.ui_body_video.number_button.setText("摄像" + object_name)


if __name__ == '__main__':
    app = QApplication()
    bv = body_vedio(None, '1')
    # print(type(bv))
    bv.ui_body_video.show()
    app.exec_()
