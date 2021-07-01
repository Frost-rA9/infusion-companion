# encoding=GBK
"""LabelHelper�ǽ�����ImageHelper�Ļ����ϵ�
    - ��Ҫ�����������һ������ͼ��ת��
    - img_label[:, :, c] = (img_block[:, :, 0] == c).astype(int)
        1. ��ÿ��ͨ����ֻ��ʾ��ǰ��Ϣ���������
        2. ʵ���Ͼ����ڶ�ֵ���������������ĵط������趨��1�����ǵ��趨��0
"""
from utils.ImageLoaderHelper.ImageHelper import ImageHelper
import numpy as np


class LabelHelper:
    def __init__(self, img_path: str, step: tuple, num_classes: int):
        # �����ȡ���Ǳ�ǩͼ
        self.image_loader_helper = ImageHelper(img_path, step)
        self.num_classes = num_classes
        self.step = step

    def read_next_label(self):
        img_block = self.image_loader_helper.read_next_block()
        if img_block.shape[:2] != self.step:
            img_block = ImageHelper.image_expand(self.step, img_block)  # ��ȡ���߽��ʱ�򲹳�ڱ���ͼ���Сͳһ

        # block_info = img_block.shape[:2]
        # img_label = np.zeros((block_info[0], block_info[1], self.num_classes))
        # for c in range(self.num_classes):
        #     img_label[:, :, c] = img_block[:, :, 0]  # ����ͨ����,����ÿһͨ����ֵ��һ�����õ�ʱ�����һ�㶼��
        return img_block


if __name__ == '__main__':
    # _label_
    l = LabelHelper("../../../../Resource/GF2_PMS1__L1A0001064454-MSS1_label_.tif",
                    (200, 200), 7)
    np.set_printoptions(threshold=np.inf)
    print(l.read_next_label().shape)
    # print(l.read_next_label())
