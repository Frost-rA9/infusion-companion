# encoding=GBK
"""BasicBlock

    - DarkNet�Ļ����в�ṹ

    1. ����1x1�ľ������ͨ����
    2. ����3x3�ľ����ȡ����
    3. �����ȡ�����Ľṹ��������������
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: list):
        super(BasicBlock, self).__init__()
        self.cov1x1 = nn.Conv2d(in_planes, planes[0],
                                kernel_size=1,
                                stride=1, padding=0,
                                bias=False)
        self.batch_norm1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.cov3x3 = nn.Conv2d(planes[0], planes[1],
                                kernel_size=3,
                                stride=1, padding=1,
                                bias=False)
        self.batch_norm2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        x = self.cov1x1(x)  # ��channelͳһ����һ��Ĵ�С
        x = self.batch_norm1(x)  # �ֲ���һ��
        x = self.relu1(x)  # ��ӷ�����

        x = self.cov3x3(x)  # ������ȡ
        x = self.batch_norm2(x)
        x = self.relu2(x)

        return x + residual  # ��Ϣ����,ǰ��ͨ��������


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512))
    b = BasicBlock(3, (5, 3))
    print(b(x).shape)
