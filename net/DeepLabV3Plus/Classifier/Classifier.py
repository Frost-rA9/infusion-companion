import torch
import torch.nn as nn
from torch.nn import functional as F
from net.DeepLabV3Plus.Classifier.ASPP import ASPP


class Classifier(nn.Module):
    def __init__(self, middle_channels, out_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(Classifier, self).__init__()
        """ project用于前四层的低特征进行处理
              - 用1x1的cov改变通道数好合并
        """
        self.project = nn.Sequential(
            nn.Conv2d(middle_channels, 48, kernel_size=1, bias=False),  # 关闭偏置，因为在BatchNorm中无用
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),  # 直接对tensor进行修改，同时增加非线性
        )
        """ aspp网络用于对骨干网络的输出进行处理
        """
        self.aspp = ASPP.ASPP(out_channels, aspp_dilate)
        """分类器用于对最后的特征图进行输出
        """
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.init_weight()

    def forward(self, feature):
        low_level_features = self.project(feature['low_level'])  # 对骨干网络的中间层进行通道数转换
        # print(low_level_features.size())  # torch.Size([1, 48, 64, 64])
        output_feature = self.aspp(feature['out'])  # 对骨干网络的输出进行ASPP处理
        output_feature = F.interpolate(output_feature, size=low_level_features.shape[2:], mode="bilinear", align_corners=False)  # 线性插值到和低特征一样的大小
        # 通道数对接后,进行最后的像素点分类,把每个像素点标注为[0-classes-1]的值
        return self.classifier(torch.cat([low_level_features, output_feature], dim=1))

    def init_weight(self):
        """自定义模块的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


