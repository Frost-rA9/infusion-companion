# encoding=GBK
"""��ɫ�ռ�ת��

    - ע��opencv����ɫͨ��ΪBGR,������RGB

    - д���������������Ƿ��֣����������ݼ���ɫ�ռ��Ƿ�RGB��
    - ��װ�������£�
        1. color_range����ͬ��ɫ�ռ����ɫ��Χ

            - ��ʽ{"��ɫ�ռ�����": (��ɫ�ռ䷶Χ����,��Χ����)}

            - RGB(��ԭɫ): [0-255],[0-255],[0-255]
            - HSV(���ֻ�ͼ��, H: ɫ��, S: ���Ͷ�, V: ����):
                - [0-179],[0-255],[0-255]
            - Lab(��ɫ֮���ŷʽ����,����Խ��,���۸й�������ɫ���ԽԶ)
                - L: �������ȣ��ϰ��ºڣ��м��ɫ
                - a: �����Һ�
                - b: һ�˴���,һ�˴���
            - �Ҷ�ͼ: Y = 0.299*R + 0.587*G + 0.114*B
                - ÿ������[0-255]
"""
"""֪ʶ�㲹��
    1. Ϊʲô��HSVͼ�������RGB
        - HSV�ռ��ܷǳ�ֱ�۵ı��ɫ�ʵ�������ɫ�����Լ����޳̶�
        - Ҳ���Ƿ�����ɫ�ĶԱ�(�ر����޶���Χ��)
    2. ps��HSV��Χ��opencv�б�
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
        color_range = self.color_range.get(space_name, None)  # ���Է���ֵ��Ҫ�ж�
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
        """Ҳ������ν��ͼ���ɰ�(mask)���ҳ�ָ����Χ�ڵ������������Χ��Ϊ1�����Ǿ���0��
           Ч������ͼ���ֵ��
        :param low_range: ��ɫ������, [x,x,x..] ����channel���
        :param upper_range: ͬ����ɫ����
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
