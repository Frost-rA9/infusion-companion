# encoding=GBK
"""LoadWithBlock

    - ���÷���������޷�ʹ���ĸ�GPU�£��ڴ治��ʱʹ��

    - ע���״��ȡ��������ndarray

    1. ���ڸ����ڴ�ֿ飬��ͬһ��������ѭ��
    2. ����block_numberʵ�ʾ��Ǹ����Լ����ڴ����
        - ��ȡ�������ļ������ѵ��
"""

from utils.ImageLoaderHelper.ImageHelper import ImageHelper
from PIL import Image
import numpy as np


class LoadWithBlock:
    def __init__(self,
                 img_path: str,
                 step: tuple,
                 block_number: int):
        self.img_path = img_path
        self.step = step
        self.block_number = block_number
        self.helper = ImageHelper(img_path, step)

        self.block_list = []

    def read_block(self):
        """ѭ����ȡblock_number�Ŀ�,�����ض�ȡ�б�"""
        self.block_list.clear()
        for i in range(self.block_number):
            self.block_list.append(self.helper.read_next_block())
        return self.block_list


if __name__ == '__main__':
    block = LoadWithBlock("../../Resource/CAER-S/Train/Anger/0001.png",
                          step=(200, 200),
                          block_number=3)
    print(len(block.read_block()))


