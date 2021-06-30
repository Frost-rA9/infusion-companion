import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ASPPConv(nn.Sequential):
    """空洞卷积
    """

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    # def forward(self, x):
    #     size = x.shape[-2:]
    #     # print(x.size())  # torch.Size([1, 256, 128, 128])
    #     if x.shape[0] > 1:
    #         super(ASPPPooling, self).__init__(*self.module_batch_over_1)
    #     else:
    #         super(ASPPPooling, self).__init__(*self.module_batch_with_1)
    #     x = super(ASPPPooling, self).forward(x.cpu())
    #     return F.interpolate(x, size=size, mode='bilinear', align_corners=False).to(device)


if __name__ == '__main__':
    x = torch.randn((2, 512, 128, 128))
    a = ASPPPooling(512, 256, 1)
    x = a(x)
    print(x.size())
