# encoding=GBK
"""ʾ������н��ͳ��

    - �ؼ�
        1. �ؼ�����
        2. �ؼ���Ϣ����
        3. �¼�����
"""
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox


def handleCalc():
    info = text_edit.toPlainText()
    QMessageBox.about(window, 'ͳ�ƽ��', info)  # ����֪ͨ�Ի���


app = QApplication([])  # ����ͼ�ν���ĵײ㹦�ܣ���ʼ��������Ϣ�ȣ��ɷ���Ϣ��

window = QMainWindow()
window.resize(500, 400)
window.move(300, 310)  # ��������Ļ��λ��, ����300������310
window.setWindowTitle("н��ͳ��")

text_edit = QPlainTextEdit(window)  # ������ڵĸ�����
text_edit.setPlaceholderText("��������Ϣ��")  # ��ʾ��Ϣ��û����Ϣʱ��ʾ
text_edit.move(10, 25)  # ��Ը����ڵ�λ��
text_edit.resize(300, 350)

button = QPushButton('ͳ��', window)
button.move(330, 120)
button.clicked.connect(handleCalc)  # ��signal(clicked)���ӵ�SLOT(handleCalc)�Ӷ��󶨲���

window.show()  # ִ�����ϲ��show������ʾȫ������
app.exec_()  # �����¼�����ѭ��
