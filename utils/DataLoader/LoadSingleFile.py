# encoding=GBK
"""LoadSingleFile

    - ����ͼ���С��С������ֱ�Ӵ��������ļ�

    - ����û�ÿ�״��ȡ������ʹ��PIL��ȡ

    - ��������:
        1. train_path, test_path:
            - ����·������·�����з����ͼƬ����
        2. is_train:
            - True��ֻ����train·���µ�����
            - False, ֻ����test·���µ�����
        3. img_list:
            - ��ŵ�ַ������ӳ��
            - ��ʽ[(address, classes), ..]
        4. trans:
            - �ڶ�ȡ������֮ǰϣ�����е�ת��
            - ���Ϊ��,����Ҫ�Լ�ת��

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
                 trans = None,
                 resize=False):

        # 0 anger ����
        # 1 disgust ���
        # 2 fear �־�
        # 3 happy ����
        # 4 sad ����
        # 5 surprised ����
        # 6 normal ����
        self.train_path = train_path
        self.test_path = test_path
        self.is_train = is_train
        self.img_list = []
        self.img_data_load()
        self.trans = trans
        self.resize = resize

    def img_data_load(self):
        """�������е�Ŀ¼, ��������"""

        if self.is_train:
            d = os.listdir(self.train_path)
            data_path = self.train_path
        else:
            d = os.listdir(self.test_path)
            data_path = self.test_path

        for file_dir in d:
            class_number = file_dir
            absolute_path = data_path + '/' + file_dir
            for file_name in os.listdir(absolute_path):
                file_path = absolute_path + "/" + file_name
                self.img_list.append((file_path, int(class_number)))
    def get_num_classes(self):
        return 7

    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = Image.open(img_path)
        if self.resize:
            img=img.resize((400, 400),Image.ANTIALIAS)
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

    img, label = l.__getitem__(i)

    print(img, label)
    print(type(img), type(label))
    print(l.__len__())