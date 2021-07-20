# encoding=GBK
"""LoadSingleFile

    - 由于图像大小较小，所以直接传回整个文件

    - 由于没用块状读取，所以使用PIL读取

    - 参数解释:
        1. train_path, test_path:
            - 绝对路径，此路径下有分类的图片即可
        2. is_train:
            - True，只导入train路径下的内容
            - False, 只导入test路径下的内容
        3. img_list:
            - 存放地址与类别的映射
            - 格式[(address, classes), ..]
        4. trans:
            - 在读取出数据之前希望进行的转换
            - 如果为空,则需要自己转换

"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch
import cv2


class LoadSingleFile(Dataset):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 is_train: bool,
                 trans=None,
                 resize=True):
        # 0 anger 生气
        # 1 disgust 厌恶
        # 2 fear 恐惧
        # 3 happy 开心
        # 4 sad 伤心
        # 5 surprised 惊讶
        # 6 normal 中性
        self.expression_dict = {
            "happy":0,  # 痛苦
            "sad":1,  # 不痛苦
        }
        self.google_expression_dict = {
            "upper": 0,  # 痛苦
            "lower": 1,  # 不痛苦
        }
        self.train_path = train_path
        self.test_path = test_path
        self.is_train = is_train
        self.img_list = []
        self.img_data_load()
        self.trans = trans
        self.resize = resize

    def img_data_load(self, is_googlenet: bool=False):
        """遍历所有的目录, 并添加类别"""

        if self.is_train:
            d = os.listdir(self.train_path)
            data_path = self.train_path
        else:
            d = os.listdir(self.test_path)
            data_path = self.test_path

        for file_dir in d:
            if is_googlenet:
                class_number = self.google_expression_dict[file_dir]
            else:
                class_number = self.expression_dict[file_dir]
            absolute_path = data_path + '/' + file_dir
            for file_name in os.listdir(absolute_path):
                file_path = absolute_path + "/" + file_name
                self.img_list.append((file_path, class_number))

    def get_num_classes(self):
        return 2

    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = Image.open(img_path)
        if self.resize:
            img = img.resize((48, 48), Image.ANTIALIAS)
        if self.trans:
            # img=np.array(img)
            # img_expend=np.zeros((299, 299), dtype=np.uint8)
            # img_expend[:48,:48]=img
            # img=img_expend
            # img = Image.fromarray(img)
            img = self.trans(img)

        return img, torch.tensor(label)

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    l = LoadSingleFile(train_path="../../Resource/CAER-S/train",
                       test_path="../../Resource/CAER-S/test",
                       is_train=False,
                       trans=trans)

    img, label = l.__getitem__(0)

    print(img, label)
    print(type(img), type(label))
    print(l.__len__())
