# encoding=GBK
"""ͼ�������
    - ���ڻ��һ��ͼ���е����ݿ�����и����������ά�����ֹ�ϵ
    1. ÿ�ε��õ�����ļ�ʱ����ȡ��һ��λ�õ�ͼ��
    2. ���ڻ���ѭ��������ÿ�ζ�ȡ��ͼ��ĩβʱ���Զ����ö�ȡλ��

    - ���ڿ�״��ȡ�ı߽��ܻ��б߽�ȱʧ��������Ҫ��ȫ
    1. �������Ǹ��ƿ���Ϣ����һ����С�ոյ���step��zeros����
"""

from utils.ImageLoaderHelper.ImageBlockHelper import ImageBlockHelper
import numpy as np


class ImageHelper:
    def __init__(self, img_path, step: tuple):
        self.img_path = img_path
        self.step = step
        self.block_helper = ImageBlockHelper(self.img_path, self.step)
        self.iterator = iter(self.block_helper.read_data_loop())

    @staticmethod
    def image_expand(step: tuple, block: np.ndarray):
        shape = block.shape
        block_zero = np.zeros((step[0], step[1], shape[2])).astype(np.uint8)
        block_zero[:shape[0], :shape[1]] = block[:, :]
        return block_zero

    def read_next_block(self):
        img = next(self.iterator)
        if img.shape[:2] != self.step:
            img = ImageHelper.image_expand(self.step, img)
            # print(img.shape)
        return img


if __name__ == '__main__':
    i = ImageHelper("../../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif",
                    (224, 224))
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    # ����read_next_block
    while True:
        i.read_next_block()

    # ����img_expand
    # img = np.ones((3, 3, 3))
    # print(i.image_expand((5, 5), img))
