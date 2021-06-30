# encoding=GBK
"""ExpandMethod
    - 通过一些初始化步骤，最终返回生成的transforms.Compose
"""
import numpy as np
from torchvision import transforms


class ExpandMethod:
    def __init__(self):
        self.param_list = []
        self.init_param_list()

    def get_transform(self):
        return transforms.Compose(self.param_list)

    def init_param_list(self):
        self.add_to_PIL()
        self.add_random_vertical_flip()
        self.add_random_horizontal_flip()
        self.add_color_jitter()
        self.add_to_tensor()

    def add_to_PIL(self):
        self.param_list.append(transforms.ToPILImage())

    def add_random_horizontal_flip(self):
        p = np.random.choice([0, 1])  # 0，1中随机选一个
        self.param_list.append(transforms.RandomHorizontalFlip(p))

    def add_random_vertical_flip(self):
        p = np.random.choice([0, 1])
        self.param_list.append(transforms.RandomVerticalFlip(p))

    def add_to_tensor(self):
        self.param_list.append(transforms.ToTensor())

    def add_color_jitter(self):
        bright_change = np.random.random()  # 随机亮度
        hue_change = abs(np.random.random() - 0.5)  # 随机色相 [-0.5, 0.5]
        contrast_change = np.random.random()  # 随机对比度
        saturation_change = np.random.random()  # 随机饱和度

        self.param_list.append(transforms.ColorJitter(brightness=bright_change,
                                                      contrast=contrast_change,
                                                      saturation=saturation_change,
                                                      hue=hue_change
                                                      ))


if __name__ == '__main__':
    e = ExpandMethod()
    trans = e.get_transform()
    from logical.utils.ImageLoaderHelper.ImageHelper import ImageHelper
    i = ImageHelper("../../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif",
                    (224, 224))
    img = i.read_next_block()
    for t in trans.transforms:
        img = t(img)
    print(img.shape)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    print(img.shape)
    import cv2 as cv
    cv.imshow("img", img)
    cv.waitKey(0)