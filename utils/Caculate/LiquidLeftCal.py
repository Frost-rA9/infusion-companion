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
        self.cur_img_data = list(count)

    def get_cur_liquid(self, img: np.ndarray):
        self.cal_img(img)  # ���ص�ͳ��
        # print(self.cur_img_data)
        return self.cur_img_data[2] / (self.cur_img_data[1] + self.cur_img_data[2])


    @staticmethod
    def predict_show(img: np.ndarray):
        """չʾ��Ԥ������Ľ����������ʱ������"""
        color_empty, color_full = (0, 0, 128), (0, 128, 0)
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # ��3ͨ��
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         b, g, r = img[i, j]
        #         if [b, g, r] == [1, 1, 1]:
        #             img[i, j] = color_empty
        #         elif [b, g, r] == [2, 2, 2]:
        #             img[i, j] = color_full
        #         else:
        #             pass
        return img * 100  # ���Ч��


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