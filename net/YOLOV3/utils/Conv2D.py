# encoding=GBK
"""Conv2D

    - �͹Ǹ�����ĵ�0��������ṹ����
        - �仯��ֻ�в��̶��ľ������ͨ����
        - ������Ӧ3����������������

    - ��װ���裺
        1. conv
        2. batch_norm
        3. LeaKyRelU
"""
import torch.nn as nn    
from collections import OrderedDict


class Conv2D:
    @staticmethod
    def conv2d(filter_in, filter_out, kernel_size):
        pad = (kernel_size - 1) // 2 if kernel_size else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))


if __name__ == '__main__':
    # img = torch.randn((1, 3, 200, 200))
    out = Conv2D.conv2d(3, 10, 3)
    print(out)