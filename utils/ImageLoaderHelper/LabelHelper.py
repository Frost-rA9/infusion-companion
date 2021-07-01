# encoding=GBK
"""LabelHelper是建立在ImageHelper的基础上的
    - 主要功能是添加了一个特征图的转换
    - img_label[:, :, c] = (img_block[:, :, 0] == c).astype(int)
        1. 在每个通道上只显示当前信息的物体类别
        2. 实际上就是在二值化，把是这个物体的地方概率设定成1，不是的设定成0
"""
from utils.ImageLoaderHelper.ImageHelper import ImageHelper
import numpy as np


class LabelHelper:
    def __init__(self, img_path: str, step: tuple, num_classes: int):
        # 这里读取的是标签图
        self.image_loader_helper = ImageHelper(img_path, step)
        self.num_classes = num_classes
        self.step = step

    def read_next_label(self):
        img_block = self.image_loader_helper.read_next_block()
        if img_block.shape[:2] != self.step:
            img_block = ImageHelper.image_expand(self.step, img_block)  # 读取到边界的时候补充黑边让图像大小统一

        # block_info = img_block.shape[:2]
        # img_label = np.zeros((block_info[0], block_info[1], self.num_classes))
        # for c in range(self.num_classes):
        #     img_label[:, :, c] = img_block[:, :, 0]  # 增加通道数,由于每一通道的值是一样的用的时候随便一层都行
        return img_block


if __name__ == '__main__':
    # _label_
    l = LabelHelper("../../../../Resource/GF2_PMS1__L1A0001064454-MSS1_label_.tif",
                    (200, 200), 7)
    np.set_printoptions(threshold=np.inf)
    print(l.read_next_label().shape)
    # print(l.read_next_label())
