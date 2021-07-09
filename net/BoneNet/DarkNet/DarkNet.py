# encoding=GBK
"""DarkNet

    - DarkNet����������

    - �������������Ϣ

    - ע��DarkNet��ͼ��ֻ����������
"""
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from net.BoneNet.DarkNet.utils.BasicBlock import BasicBlock


class DarkNet(nn.Module):
    def __init__(self,
                 layers: list):  # layers����ÿ����Ҫ������
        super(DarkNet, self).__init__()
        self.inplanes = 32

        # 0. 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 1. 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 2. 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 3. 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 4. 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 5. 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        # ÿ�����ͨ���ı仯
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        # 6. Ȩ�س�ʼ��
        self.init_weight()

    def forward(self, x):
        # 1. ��������
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 2. ����5��ṹ
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []

        # 1. ÿ��layers������������һ������Ϊ2��3x3��������²���
        #   - ����Ϊ2���Խ���ͼ���С���൱������1/2��ͼ���С
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 2. ѭ����Ӳв�ṹ
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))


if __name__ == '__main__':
    net = DarkNet([1, 2, 8, 8, 4])  # ��ֵ��������
    x = torch.randn((1, 3, 416, 416))
    x3, x4, x5 = net(x)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)