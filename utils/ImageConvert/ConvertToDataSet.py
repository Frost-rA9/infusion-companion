# encoding=GBK
"""ConvertToDataSet

    - 转变某个文件夹内的全部图像到一个合理的大小

    - 包含以下步骤：
        1. resize
        2. to gray
"""
import os
import cv2 as cv


class ConvertToDataSet:
    def __init__(self, convert_dir, des_dir):
        self.img_path = []
        self.img_shape = None

        self.convert_dir = convert_dir
        self.des_dir = des_dir

    def get_all_img(self):
        for file in os.listdir(self.convert_dir):
            self.img_path.append(self.convert_dir + file)

    def cal_all_img(self):
        height, width = 0, 0
        for img_file in self.img_path:
            img = cv.imread(img_file)
            h, w = img.shape[:2]
            width += w
            height += h
        self.img_shape = (height / len(self.img_path), width / len(self.img_path))

        if abs(self.img_shape[0] - self.img_shape[1]) / self.img_shape[0] < 10:
            self.img_shape = (int(self.img_shape[0]), int(self.img_shape[0]))
        else:
            self.img_shape = (int(self.img_shape[0]), int(self.img_shape[1]))
        print(self.img_shape)

    def img_convert(self, img_shape: tuple = None):
        if img_shape:
            height, width = img_shape
        else:
            height, width = self.img_shape
        for img_file in self.img_path:
            img = cv.imread(img_file)
            img = cv.resize(img, dsize=(height, width))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            img_name = img_file.split("/")
            if not img_name:
                img_name = img_file.split("\\")
            img_name = img_name[-1]

            new_file_path = self.des_dir + img_name
            cv.imwrite(new_file_path, img)


if __name__ == '__main__':
    convert = ConvertToDataSet("F:/DataSet/Expression/painful/",
                               "F:/DataSet/Expression/des/")
    convert.get_all_img()
    convert.cal_all_img()
    convert.img_convert(img_shape=(224, 224))
