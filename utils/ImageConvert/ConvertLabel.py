# encoding=GBK
"""ConvertLabel

    ���ݴ�����ֵ�ֵ���ѱ��ͼ�������ͼת�������ͼ

    - Ĭ�ϵ��ֵ���Կ�������ֵ���б��
    - �ֵ��ʽ {����ֵ: tuple(r,g,b)�����: int}
"""
import cv2 as cv
import os
import threading


class ConvertThread(threading.Thread):
    def __init__(self, file_list: list, convert_dict: dict):
        super(ConvertThread, self).__init__()
        self.file_list = file_list
        self.convert_dict = convert_dict

    def run(self):
        for file in self.file_list:
            self.to_label(file)

    def to_label(self, img_path: str):
        img = cv.imread(img_path)  # BGR
        img_height, img_width, _ = img.shape
        for height in range(img_height):
            for width in range(img_width):
                b, g, r = tuple(img[height, width])
                label = self.convert_dict[(r, g, b)]
                img[height, width] = label  # ÿ������ֵע�ͳɱ�ǩ
        cv.imwrite(img_path[:-4]+"_.tif", img)


class ConvertLabel:
    def __init__(self, convert_dict: dict = None, convert_dir: str = None):
        super().__init__()
        if not convert_dict is None:
            self.convert_dict = convert_dict
        else:
            self.convert_dict = {
                (0, 0, 0):       0,  # black - background - ����
                (200, 0, 0):     1,  # industrial land - ��ҵ�õ�
                (250, 0, 150):   2,  # urban residential - סլ�õ�
                (200, 150, 150): 3,  # rural residential - ũ������õ�
                (250, 150, 150): 4,  # traffic land - ��ͨ�õ�
                (0, 200, 0):     5,  # paddy field - ˮ��
                (150, 250, 0):   6,  # irrigated land - ��ȵ�
                (150, 200, 150): 7,  # dry cropland - ������
                (200, 0, 200):   8,  # garden land - ԰��
                (150, 0, 250):   9,  # arbor forest - ��ľ��
                (150, 150, 250): 10,  # shrub land - ��ľ��
                (250, 200, 0):   11,  # natural meadow - ��Ȼ�ݵ�
                (200, 200, 0):   12,  # artificial meadow - �˹��ݵ�
                (0, 0, 200):     13,  # river - ����
                (0, 150, 200):   14,  # lake - ����
                (0, 200, 250):   15,  # pond - ��ָ�˹���
            }

        self.convert_dir = convert_dir
        self.file_path = []  # �����洢����Ŀ¼ʱ����ļ���
        if not convert_dir is None:
            self.read_file(convert_dir)

    def read_file(self, path):
        """��������������"""
        dir = os.listdir(path)
        absolute_path = path + "/"
        for file in dir:
            self.file_path.append(absolute_path+file)
        self.img_convert()

    def img_convert(self):
        """���ö��߳̽���ת��"""
        for file in self.file_path:
            thread = ConvertThread([file], self.convert_dict)
            thread.start()


if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    c = ConvertLabel(convert_dir="H:/DataSet/Fine Land-cover Classification_15classes/label_15classes")
    path = "H:/DataSet/Fine Land-cover Classification_15classes/label_15classes/GF2_PMS1__L1A0001064454-MSS1_label.tif"
    # c.to_label(path)