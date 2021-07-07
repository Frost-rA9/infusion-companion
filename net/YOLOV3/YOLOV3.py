# encoding=GBK
"""YOLOV3

    - 封装YOLO3的回归与分类部分

    - BoneNet:
        - 三个有效特征层
            - 52,52,256
            - 26,26,512
            - 13,13,1024

    - final_out_filter:
        - 具体含义
        - 3(三个框) * ( 1(里面有没有物体) + 4(left,top,width,height) + num_classes)

    - YOLO3:
        - 对骨干网络的提取的特征进行进一步提取
        - 然后对提取结果进行回归与分类
        - 对于最后两层还要把提取的特征与上级提取结果合并
"""

import torch
import torch.nn as nn

from net.BoneNet.DarkNet.DarkNet53 import DarkNet53
from net.YOLOV3.utils.Conv2D import Conv2D
from net.YOLOV3.utils.Layers import Layers


class YOLOV3(nn.Module):
    def __init__(self, anchor, num_classes):
        super(YOLOV3, self).__init__()
        self.backbone = DarkNet53.get_model()

        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        # 1. 13,13,1024的分类层
        final_out_filter0 = len(anchor[0]) * (5 + num_classes)
        self.last_layer0 = Layers.make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        # 2. 26,26,512的分类层
        final_out_filter1 = len(anchor[1]) * (5 + num_classes)
        self.last_layer1_conv = Conv2D.conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = Layers.make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        # 3. 52,52,256的分类层
        final_out_filter2 = len(anchor[2]) * (5 + num_classes)
        self.last_layer2_conv = Conv2D.conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = Layers.make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)

    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        # 1. 获得三个有效特征层
        x2, x1, x0 = self.backbone(x)

        # 2. 第一个特征层
        # 2.1 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 2.2 主备第二个特征层的前置层
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 2.3 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)

        # 3. 第二个特征层
        # 3.1 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 3.2 准备第三个特征层的前置层
        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 3.3 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        # 4. 第3个特征层
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2


if __name__ == '__main__':
    import numpy as np
    torch.set_printoptions(threshold=np.inf)
    y = YOLOV3([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 3)
    x = torch.randn((1, 3, 416, 416))
    x1, x2, x3 = y(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x1)

