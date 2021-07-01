# encoding=GBK
"""颜色空间转换

    - 注意opencv的颜色通道为BGR,而不是RGB

    - 写这个工具类的作用是发现，比赛的数据集颜色空间是非RGB的
    - 封装内容如下：
        1. color_range：不同颜色空间的颜色范围

            - 格式{"颜色空间名字": (颜色空间范围下限,范围上限)}

            - RGB(三原色): [0-255],[0-255],[0-255]
            - HSV(数字化图像, H: 色调, S: 饱和度, V: 亮度):
                - [0-179],[0-255],[0-255]
            - Lab(颜色之间的欧式距离,距离越大,人眼感官两种颜色差距越远)
                - L: 像素亮度，上百下黑，中间灰色
                - a: 左绿右红
                - b: 一端纯蓝,一端纯黄
            - 灰度图: Y = 0.299*R + 0.587*G + 0.114*B
                - 每个像素[0-255]
"""
"""知识点补充
    1. 为什么用HSV图像而不是RGB
        - HSV空间能非常直观的表达色彩的阴暗，色调，以及鲜艳程度
        - 也就是方便颜色的对比(特别在限定范围后)
    2. ps的HSV范围与opencv有别
        - HSV: [0-366], [0-1], [0-1]
"""

import cv2 as cv
import numpy as np
import warnings


class ConvertColorSpace:
    def __init__(self):
        self.color_range = {
            "RGB": ([0, 0, 0], [255, 255, 255]),
            "HSV": ([0, 0, 0], [179, 255, 255]),
            "GRAY": ([0, 0, 0], [255, 255, 255])
        }

    def get_color_range(self, space_name: str):
        color_range = self.color_range.get(space_name, None)  # 所以返回值需要判断
        if not color_range:
            warnings.warn("ConvertColorSpace: color space is not exist")
        return color_range

    @staticmethod
    def bgr_and_hsv(to_hsv: bool, img: np.ndarray):
        """
        :param img: wait be change
        :param to_hsv: True, bgr-> hsv; False, hsv->gbr
        :return: np.ndarray
        """
        if to_hsv:
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        else:
            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        return img

    @staticmethod
    def bgr_and_gray(to_gray: bool, img: np.ndarray):
        if to_gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def get_pixel_in_range(low_range: list, upper_range: list, img: np.ndarray):
        """也就是所谓的图像蒙板(mask)，找出指定范围内的像素在这个范围内为1，不是就是0，
           效果就是图像二值化
        :param low_range: 颜色的下限, [x,x,x..] 几个channel填几个
        :param upper_range: 同理颜色上限
        :return: np.ndarray
        """
        shape = img.shape[-1]  # channel
        not_match_flag = False
        if len(low_range) != shape or len(upper_range) != shape:
            warnings.warn("ConvertColorSpace: low or upper channels is not match")
            not_match_flag = True
        if not_match_flag:
            low_range.extend([low_range[0] for i in range(shape - len(low_range))])
            upper_range.extend([upper_range[0] for i in range(shape - len(upper_range))])
        mask = cv.inRange(img, np.array(low_range), np.array(upper_range))
        return mask


if __name__ == '__main__':
    img_path = "F:/DataSet/A3/E117D5_N34D2_20180204_GF2_DOM_4_fus/E117D5_N34D2_20180204_GF2_DOM_4_fus.tif"
    start, end, step = (20000, 20000), (100000, 100000), (400, 400)
    from utils.ImageRead.Block import BlockImage

    Block = BlockImage(img_path, start, end, step)
    np.set_printoptions(threshold=np.inf)
    for img in Block.read_data():
        img = img.astype(np.uint8)
        b, g, r, near = [img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]]
        # mask = ConvertColorSpace.get_pixel_in_range([0], [255], img)
        # mask = ConvertColorSpace.bgr_and_hsv(False, img.astype(np.uint8)[:,:,:3])
        cv.imshow("img", cv.merge([b, g, r]))
        cv.waitKey(0)
