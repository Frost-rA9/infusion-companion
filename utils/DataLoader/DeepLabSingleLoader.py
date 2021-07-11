# encoding=GBK
"""DeepLabSingleLoader

    - 直接将一整个训练文件传输到网络中

"""
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DeepLabSingleLoader(Dataset):
    def __init__(self, file_path, train_trans, test_trans, num_classes, is_train: bool):
        super(DeepLabSingleLoader, self).__init__()
        self.data_list = []
        self.train_trans = train_trans
        self.test_trans = test_trans
        self.num_classes = num_classes
        self.is_train = is_train

        self.read_data(file_path)

    def read_data(self, file_path: str):
        d = os.listdir(file_path)
        for item in d:
            sub_dir = os.listdir(file_path + item)
            if 'img.png' in sub_dir and 'label.png' in sub_dir:
                img = file_path + item + "/" + 'img.png'
                label = file_path + item + "/" + 'label.png'
                self.data_list.append((img, label))

    def to_one_hot(self, img):
        width, height = img.size
        seg_labels = np.zeros((height, width, self.num_classes))
        img = np.array(img)
        for c in range(self.num_classes):
            seg_labels[:, :, c] = (img[:, :] == c)
        # seg_labels = np.reshape(seg_labels, (-1, self.num_classes))
        img = Image.fromarray(seg_labels.astype(np.uint8))
        return img

    def __getitem__(self, index):
        img_path, label_path = self.data_list[index]
        img = Image.open(img_path)
        label = Image.open(label_path)
        if self.is_train:
            label = self.to_one_hot(label)
        return self.train_trans(img), self.test_trans(label)
        # return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    file_path = "F:/DataSet/bottle/segmentation/dir_json/test/"
    d = DeepLabSingleLoader(file_path, None, None, 3, True)
    np.set_printoptions(threshold=np.inf)
    img, label = d.__getitem__(0)
    print(label.size)
    label = np.array(label)
    print(label.shape)
