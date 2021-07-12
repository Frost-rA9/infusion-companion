# encoding=GBK
"""ExpandMethod
    - 通过一些初始化步骤，最终返回生成的transforms.Compose

    - 关于使用：
        1. 如果没有特别需要直接调用get_transform就可以拿到整理好的trans
            - 随机的图像变换
            - ToTensor
            - 随机图像裁剪
        2. 如果对结果不满意：
            - 首先用clear_list清除list
            - 然后调用函数往里加
            - 加完用get_transform得到生成的trans
        3. 注意顺序，这些增广有一定的前后顺序，否则报错
"""
"""常见的图像增广技术
简单的分类如下：
标准数据增广：泛指深度学习前期或更早期的一些常用数据增广方法。
    - 数据IO；包含ToTensor、ToPILImage、PILToTensor等
    - 图像镜像类：包含RandomHorizontalFlip、RandomVerticalFlip、RandomRotation、Transpose等
    - 颜色空间变换：ColoeJitter、Grayscale、
    - 图像裁剪缩放类：Resize、CenterCrop、RandomResizedCrop、TenCrop、FiveCrop等
    - 图像线性变换类：LinearTransform、RandomRotation、RandomPerspective
    - 图像变换类：泛指基于NAS搜索到的一组变换组合，包含AutoAugment、RandAugment、Fast AutoAugment、Faster AutoAugment、Greedy Augment等；
    - 图像裁剪类：泛指深度学习时代提出的一些类似dropout的数据增广方法，包含CutOut、RandErasing、HideAndSeek、GridMask等；
    - 图像混叠类：泛指在batch层面进行的操作，包含Mixup、Cutmix、Fmix等

图像变换类：
    - 根据一定的策略从基本的转换方式中选取机种进行自动的数据增强
    - 本次使用RandAugment, ToTensor前使用
图像裁剪类：
    - 随机遮挡图像，训练模型在遮挡数据上的泛化能力
    - 本次使用RandErasing, ToTensor后使用
图像混叠类
     - 图像混叠主要对 Batch 后的数据进行混合，
     - 这类数据增广方式不仅对输入进行调整，同时还进行lable的调整以及损失函数的调整。
"""
import numpy as np
from torchvision import transforms
from utils.ImageExpand.RandAugment import Rand_Augment as RandAugment
from utils.ImageExpand.RandomErasing import transforms as RandomErasing


class ExpandMethod:
    def __init__(self):
        self.param_list = []
        self.init_param_list()

    def get_transform(self):
        # 得到trans
        return transforms.Compose(self.param_list)

    def clear_list(self):
        self.param_list.clear()

    def init_param_list(self):
        # self.add_to_PIL()
        # self.add_resize()
        self.add_gray_scale()
        self.add_rand_augment()

        self.add_to_tensor()
        self.add_rand_erasing()
        # self.add_scale()
        self.add_Normalize()
    def add_Normalize(self):
        self.param_list.append( transforms.Normalize(
            mean=(0.5060045,0.5060045,0.5060045),std= (0.21260363,0.21260363,0.21260363)))
    def add_gray_scale(self):
        self.param_list.append(transforms.Grayscale(3))

    def add_scale(self):
        self.param_list.append(transforms.Scale((400, 712)))

    def add_resize(self):
        self.param_list.append(transforms.Resize((400, 536)))

    def add_rand_augment(self):
        self.param_list.append(RandAugment.Rand_Augment())

    def add_rand_erasing(self):
        self.param_list.append(RandomErasing.RandomErasing())

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
    from PIL import Image

    e = ExpandMethod()
    trans = e.get_transform()
    from utils.ImageLoaderHelper.ImageHelper import ImageHelper

    i = ImageHelper("../../Resource/CAER-S/Train/Anger/0001.png",
                    (224, 224))
    img = i.read_next_block()
    # img = Image.open("../../Resource/CAER-S/Train/Anger/0001.png")
    img = Image.fromarray(img)
    for t in trans.transforms:
        img = t(img)
    print(img.shape)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    print(img.shape)
    import cv2 as cv

    cv.imshow("img", img)
    cv.waitKey(0)
