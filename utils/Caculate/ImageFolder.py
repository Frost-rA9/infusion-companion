import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from utils.DataLoader.LoadSingleFile import LoadSingleFile

class ImageFolder():
     def __init__(self ):
         self.batch_size = 1
         self.shuffle = True

     def getStat(self,is_train):
         '''
         Compute mean and variance for  data
         :param train_data: 自定义类Dataset(或ImageFolder即可)
         :return: (mean, std)
         '''
         trans=transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),])
         dataset = LoadSingleFile(  train_path='../../../Expression/train',
                                    test_path='../../../Expression/test',
                                    is_train=is_train,
                                    trans=trans,
                                    resize=True )
         print('Compute mean and variance for trai+ning data.')
         print(len(dataset))
         loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle )
         mean = torch.zeros(3)
         std = torch.zeros(3)
         for X, _ in loader:
             for d in range(3):
                 mean[d] += X[:, d, :, :].mean()
                 std[d] += X[:, d, :, :].std()
         mean.div_(len(dataset))
         std.div_(len(dataset))
         return list(mean.numpy()), list(std.numpy())

     def get_transform(self,is_train=True):
         mean,var=self.getStat(is_train)
         transform_list = [
             transforms.Resize((400, 400)),
             transforms.CenterCrop(448),
             transforms.ToTensor(),
             transforms.Normalize(mean, var)
         ]
         return transforms.Compose(transform_list)


if __name__ == '__main__':
        folder= ImageFolder()
        mean,std=folder.getStat(is_train=False)
        print(mean,std)

