# encoding=GBK
"""Layers

    - �������ڽ��з����ʱ����Ҫ7�ξ��
        - ǰ5����ȡ����
        - ��������ȡyolo�����Ԥ����
"""
import torch.nn as nn
from net.YOLOV3.utils.Conv2D import Conv2D


class Layers:
    @staticmethod
    def make_last_layers(filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            Conv2D.conv2d(in_filters, filters_list[0], 1),
            Conv2D.conv2d(filters_list[0], filters_list[1], 3),
            Conv2D.conv2d(filters_list[1], filters_list[0], 1),
            Conv2D.conv2d(filters_list[0], filters_list[1], 3),
            Conv2D.conv2d(filters_list[1], filters_list[0], 1),
            Conv2D.conv2d(filters_list[0], filters_list[1], 3),
            nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                      stride=1, padding=0, bias=True)
        ])
        return m