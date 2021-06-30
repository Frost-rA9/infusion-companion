import torch
import torch.nn as nn
from torch.nn import functional as F
from net.DeepLabV3Plus.Classifier.ASPP import ASPPComp


class ASPP(nn.Module):
    # ASPP结构主要是对骨干网络的特征输出进行处理
    def __init__(self, in_channels, dilation_list, out_channels=256):
        super(ASPP, self).__init__()
        self.modules = []
        # 1x1的卷积
        self.modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 不同的rate进行空洞卷积
        for rate in dilation_list:
            self.modules.append(ASPPComp.ASPPConv(in_channels, out_channels, rate))
        # 池化层
        self.modules.append(ASPPComp.ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(self.modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x = x.to(device)
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=size, mode='bilinear', align_corners=False)
        res = torch.cat(res, dim=1)  # 将ASPP结构的5个输出合并
        return self.project(res)
