# encoding=GBK
"""ExpandMethod
    - ͨ��һЩ��ʼ�����裬���շ������ɵ�transforms.Compose

    - ����ʹ�ã�
        1. ���û���ر���Ҫֱ�ӵ���get_transform�Ϳ����õ�����õ�trans
            - �����ͼ��任
            - ToTensor
            - ���ͼ��ü�
        2. ����Խ�������⣺
            - ������clear_list���list
            - Ȼ����ú��������
            - ������get_transform�õ����ɵ�trans
        3. ע��˳����Щ������һ����ǰ��˳�򣬷��򱨴�
"""
"""������ͼ�����㼼��
�򵥵ķ������£�
��׼�������㣺��ָ���ѧϰǰ�ڻ�����ڵ�һЩ�����������㷽����
    - ����IO������ToTensor��ToPILImage��PILToTensor��
    - ͼ�����ࣺ����RandomHorizontalFlip��RandomVerticalFlip��RandomRotation��Transpose��
    - ��ɫ�ռ�任��ColoeJitter��Grayscale��
    - ͼ��ü������ࣺResize��CenterCrop��RandomResizedCrop��TenCrop��FiveCrop��
    - ͼ�����Ա任�ࣺLinearTransform��RandomRotation��RandomPerspective
    - ͼ��任�ࣺ��ָ����NAS��������һ��任��ϣ�����AutoAugment��RandAugment��Fast AutoAugment��Faster AutoAugment��Greedy Augment�ȣ�
    - ͼ��ü��ࣺ��ָ���ѧϰʱ�������һЩ����dropout���������㷽��������CutOut��RandErasing��HideAndSeek��GridMask�ȣ�
    - ͼ�����ࣺ��ָ��batch������еĲ���������Mixup��Cutmix��Fmix��

ͼ��任�ࣺ
    - ����һ���Ĳ��Դӻ�����ת����ʽ��ѡȡ���ֽ����Զ���������ǿ
    - ����ʹ��RandAugment, ToTensorǰʹ��
ͼ��ü��ࣺ
    - ����ڵ�ͼ��ѵ��ģ�����ڵ������ϵķ�������
    - ����ʹ��RandErasing, ToTensor��ʹ��
ͼ������
     - ͼ������Ҫ�� Batch ������ݽ��л�ϣ�
     - �����������㷽ʽ������������е�����ͬʱ������lable�ĵ����Լ���ʧ�����ĵ�����
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
        # �õ�trans
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
        p = np.random.choice([0, 1])  # 0��1�����ѡһ��
        self.param_list.append(transforms.RandomHorizontalFlip(p))

    def add_random_vertical_flip(self):
        p = np.random.choice([0, 1])
        self.param_list.append(transforms.RandomVerticalFlip(p))

    def add_to_tensor(self):
        self.param_list.append(transforms.ToTensor())

    def add_color_jitter(self):
        bright_change = np.random.random()  # �������
        hue_change = abs(np.random.random() - 0.5)  # ���ɫ�� [-0.5, 0.5]
        contrast_change = np.random.random()  # ����Աȶ�
        saturation_change = np.random.random()  # ������Ͷ�

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
