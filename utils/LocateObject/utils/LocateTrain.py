# encoding=GBK
"""
    - ��װdlib�ķ�װ����

    - ����xml�ļ�
        1. imglab.exe -c xxx.xml img_path ����xml�ļ�
        2. imglab.exe xxx.xml ���б�ע
            - ��סshiftѡ������
            - ע�����Ϸ��ĸ��������label
        3. ��ɺ�file-save
    - ����svm�ļ�
        1. ͨ��dlib��ѵ���õ����൱��ģ�ͣ�����Ԥ��
"""

import dlib


class LocateTrain:
    def __init__(self,
                 detect_xml_path: str,
                 output_svm_name: str,
                 open_flips: bool = True,
                 c_count: int = 5,
                 thread_number: int = 8,
                 verbose: bool = True):
        # ����ѵ��������ģʽ
        self.options = dlib.simple_object_detector_training_options()
        # �������ҷ�ת
        self.options.add_left_right_image_flips = open_flips
        # ����������
        self.options.C = c_count
        # �߳�����
        self.options.num_threads = thread_number
        # �������ɭ��
        self.options.be_verbose = verbose
        # ����ѵ��
        output_svm_name = "../../Resource/svm/" + output_svm_name + ".svm"
        dlib.train_simple_object_detector(detect_xml_path, output_svm_name, self.options)


if __name__ == '__main__':
    # l = LocateTrain("../../Resource/svm_label_xml/bottle.xml", "bottle_svm", c_count=10)
    from utils.LocateObject.dlibLocate import LocateRoI
    import cv2 as cv
    svm_path = "../../Resource/svm/bottle_svm.svm"
    img_path = "../../Resource/LiquidlevelDataSet/11.jpg"
    img = cv.imread(img_path)
    L = LocateRoI(svm_path)
    L.predict_show(img, color=(0, 0, 255))



