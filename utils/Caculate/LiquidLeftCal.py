# encoding=GBK
"""LiquidLeftCal

    - Һλ����utils

    - ֻҪ����get_cur_liquid���������ʣ��Һλ

    - ˼·: �����ⷨ

        1. ͨ����λ���紫�����Ķ�λ��Ϣ����ȡROI
        2. ��ROI����������зָ�
        3. ͳ�Ʒ�0�ĵ����ͬʱ�ҳ����
            - _background_: 0
            - empty: 1
            - full: 2
        4. ͳ�Ʒ���Ϊfull�����ص���ͼ���е�ռ��

    - �������3��4����������
"""
import numpy as np
import cv2 as cv


class LiquidLeftCal:
    def __init__(self):
        self.cur_img_data = [0, 0, 0]  # background, emtpy, full

    def cal_img(self, img: np.ndarray):
        count = np.bincount(img.flatten())
        # self.cur_img_data = list(count)
        count_list = list(count)  # ע���п���û��⵽full,empty�ȵ��³��Ȼ��С
        count_list.extend([0 for i in range(len(self.cur_img_data) - len(count_list))])
        self.cur_img_data = count_list

    def get_cur_liquid(self, img: np.ndarray):
        self.cal_img(img)  # ���ص�ͳ��
        # print(self.cur_img_data)
        return self.cur_img_data[2] / (self.cur_img_data[1] + self.cur_img_data[2])

    @staticmethod
    def predict_show(img: np.ndarray):
        """չʾ��Ԥ������Ľ����������ʱ������"""
        # empty = 1, full = 2
        mask_value_1 = (img == 1)
        mask_value_2 = (img == 2)

        blue, green, red = np.zeros(img.shape[:2]), np.zeros(img.shape[:2]), np.zeros(img.shape[:2])
        green[mask_value_1] = 128
        blue[mask_value_2] = 128
        return cv.merge([blue, green, red])


if __name__ == '__main__':
    from PIL import Image

    img_path = "F:/DataSet/bottle/segmentation/dir_json/train/1_json/label.png"
    img = Image.open(img_path)  # ��ȡpģʽͼ��
    img = np.array(img)  # ��np.ndarry
    np.set_printoptions(threshold=np.inf)
    # LiquidLeftCal.predict_show(img)
    liquid = LiquidLeftCal()
    threshold = liquid.get_cur_liquid(img)
    print(threshold)

    img = liquid.predict_show(img)
    cv.imshow("img", img)
    cv.waitKey(0)
