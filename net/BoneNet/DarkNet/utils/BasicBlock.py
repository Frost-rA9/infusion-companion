# encoding=GBK
"""BasicBlock

    - DarkNet的基本残差结构

    1. 先用1x1的卷积调整通道数
    2. 在用3x3的卷积提取特征
    3. 最后将提取特征的结构与输入数据连接
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: list):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)  # 将channel统一到这一层的大小
        x = self.bn1(x)  # 局部均一化
        x = self.relu1(x)  # 添加非线性

        x = self.conv2(x)  # 特征提取
        x = self.bn2(x)
        x = self.relu2(x)

        return x + residual  # 信息连接,前后通道数不变


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512))
    b = BasicBlock(3, (5, 3))
    print(b(x).shape)
