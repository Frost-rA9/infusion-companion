# encoding=GBK
"""LiquidLeftCal

    - 液位检测的utils

    - 只要调用get_cur_liquid函数，获得剩余液位

    - 思路: 暴力解法

        1. 通过定位网络传回来的定位信息，获取ROI
        2. 将ROI送往网络进行分割
        3. 统计非0的点个数同时找出标记
            - _background_: 0
            - empty: 1
            - full: 2
        4. 统计分类为full的像素点在图像中的占比

    - 本类完成3和4这两个步骤
"""
import numpy as np
import cv2 as cv


class LiquidLeftCal:
    def __init__(self):
        self.cur_img_data = [0, 0, 0]  # background, emtpy, full

    def cal_img(self, img: np.ndarray):
        count = np.bincount(img.flatten())
        # self.cur_img_data = list(count)
        count_list = list(count)  # 注意有可能没检测到full,empty等导致长度会变小
        count_list.extend([0 for i in range(len(self.cur_img_data) - len(count_list))])
        self.cur_img_data = count_list

    def get_cur_liquid(self, img: np.ndarray):
        self.cal_img(img)  # 像素点统计
        # print(self.cur_img_data)
        return self.cur_img_data[2] / (self.cur_img_data[1] + self.cur_img_data[2])

    @staticmethod
    def predict_show(img: np.ndarray):
        """展示下预测出来的结果，开发的时候用下"""
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
    img = Image.open(img_path)  # 读取p模式图像
    img = np.array(img)  # 到np.ndarry
    np.set_printoptions(threshold=np.inf)
    # LiquidLeftCal.predict_show(img)
    liquid = LiquidLeftCal()
    threshold = liquid.get_cur_liquid(img)
    print(threshold)

    img = liquid.predict_show(img)
    cv.imshow("img", img)
    cv.waitKey(0)
