# encoding=GBK
"""图像帮助者
    - 由于会对一个图像中的数据块进行切割，建立此类来维护这种关系
    1. 每次调用到这个文件时，读取下一个位置的图像
    2. 由于会多次循环，所以每次读取到图像末尾时，自动重置读取位置

    - 由于块状读取的边界总会有边界缺失，所以需要补全
    1. 方法就是复制块信息到另一个大小刚刚等于step的zeros区域
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
    # 测试read_next_block
    while True:
        i.read_next_block()

    # 测试img_expand
    # img = np.ones((3, 3, 3))
    # print(i.image_expand((5, 5), img))
