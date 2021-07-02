# encoding=GBK
"""LoadWithBlock

    - 备用方案，如果无法使用四个GPU下，内存不足时使用

    - 注意块状读取的内容是ndarray

    1. 由于根据内存分块，且同一对象无限循环
    2. 所以block_number实际就是根据自己的内存情况
        - 读取适量的文件块进行训练
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
        """循环读取block_number的块,并返回读取列表"""
        self.block_list.clear()
        for i in range(self.block_number):
            self.block_list.append(self.helper.read_next_block())
        return self.block_list


if __name__ == '__main__':
    block = LoadWithBlock("../../Resource/CAER-S/Train/Anger/0001.png",
                          step=(200, 200),
                          block_number=3)
    print(len(block.read_block()))


