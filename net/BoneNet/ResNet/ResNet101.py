"""
    - 将打包好的模型进行再度封装
"""
from net.BoneNet.ResNet.Bottleneck import Bottleneck
from net.BoneNet.ResNet.ResNet import ResNet
import torch


class ResNet101:
    @staticmethod
    def get_res_net101(output_stride, BatchNorm, pretrained_loc, pretrained=True):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, pretrained_loc= pretrained_loc)
        return model


if __name__ == "__main__":
    model = ResNet101.get_res_net101(BatchNorm=torch.nn.BatchNorm2d, pretrained=True, output_stride=8,
                                     pretrained_loc="../../../Resource/model_data/resnet101-5d3b4d8f.pth")
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())