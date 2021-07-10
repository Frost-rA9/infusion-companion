# encoding=GBK
"""
    - �Զ����������

    - ÿ��һ��ͼ��ͽ���һ��ӳ�䣬{index, (ImageHelper, LabelHelper)}
    1. ÿ�ε��ö����е�read_next_��ȡ��һ��ͼƬ
"""

from utils.ImageLoaderHelper.ImageHelper import ImageHelper
from utils.ImageLoaderHelper.LabelHelper import LabelHelper
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class BigImageDataSet(Dataset):
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 transform,
                 step: tuple,
                 num_classes: int):
        super(BigImageDataSet, self).__init__()
        # ��¼�ļ�λ��
        self.data_path = data_path
        self.label_path = label_path

        # �ļ��Ķ�ȡ����
        self.step = step
        self.num_classes = num_classes

        # �����ͼ���ת��
        self.transform = transform

        # �����м���Ϣ
        self.img_helper = []  # [(ImageHelper, LabelHelper), .]
        self.load()  # �ļ����뿪ʼ

    def __getitem__(self, index):
        data, label = self.img_helper[index]
        data = data.read_next_block()
        label = label.read_next_label()

        return self.transform(data), self.transform(label)

    def __len__(self):
        return len(self.img_helper)

    def load(self):
        img_name, img_with_path = self.load_data()
        label_with_path = self.load_label(img_name, differ="_label_")
        for img, label in zip(img_with_path, label_with_path):
            img_helper = ImageHelper(img, self.step)
            label_helper = LabelHelper(label, self.step, self.num_classes)
            self.img_helper.append((img_helper, label_helper))

    def load_data(self):
        """��data_path���ļ����ݶ����ֵ�"""
        d = os.listdir(self.data_path)
        absolute_path = self.data_path + '/'
        temp = []
        for name in d:
            temp.append(absolute_path + name)
        return d, temp  # dir���ڸ�label�Ҷ�ӦͼƬ�Ӷ������б�˳���, temp�����ڳ��������

    def load_label(self, img_name: list, differ: str):
        d = os.listdir(self.label_path)
        absolute_path = self.label_path + "/"
        temp = []
        for name in img_name:
            if name[:-4] + differ + ".tif" in d:
                temp.append(absolute_path + name[:-4] + differ + ".tif")  # Ϊ��λ�����
        return temp


if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    # train:
    img = "H:/DataSet/train/image"
    label = "H:/DataSet/train/label"

    loader = BigImageDataSet(img, label, trans, (224, 224), 7)
    img, label = loader.__getitem__(0)
    print(img.shape, label.shape)  # torch.Size([3, 224, 224]) torch.Size([7, 224, 224])
    # for i in range(3):
    #     for j in range(224):
    #         print(label[i, j])

