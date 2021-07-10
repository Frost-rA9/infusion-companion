"""

    - 类作用：
        - 组装骨干网络还有后续的分类器网络
        - 使之称为真正的DeepLabv3plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.DeepLabV3Plus.Classifier.Classifier import Classifier
from net.BoneNet.MobileNet.mobilenetv2 import MobileNetV2
from net.BoneNet.ResNet.ResNet101 import ResNet101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepLabV3Plus(nn.Module):
    """主要就是把骨干网络返回的信息做成字典：
        { 'low_level': torch, "out": torch}
        然后把字典传给分类器进行识别
    """
    def __init__(self, bone_net, classifier: Classifier):
        super(DeepLabV3Plus, self).__init__()
        self.bone_net = bone_net
        self.classifier = classifier
        self.classifier.to(device)

    def forward(self, x):
        # 保留输入信息
        input_shape = x.shape[-2:]
        # 传递给bone_net
        output, low_level_feat = self.bone_net(x)
        # 获取特征字典
        # out torch.Size([1, 2048, 64, 64])
        # low_level torch.Size([1, 256, 128, 128])
        features = {"low_level": low_level_feat, "out": output}
        # 交给分类器得出结果
        x = self.classifier(features)
        # 上采样到原来图像的大小
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = MobileNetV2(num_classes=7)
    model = ResNet101.get_res_net101(BatchNorm=torch.nn.BatchNorm2d, pretrained=True, output_stride=8,
                                     pretrained_loc="../../Resource/model_data/resnet101-5d3b4d8f.pth")
    # model.to(device)
    classifier = Classifier(middle_channels=256, out_channels=2048, num_classes=7)
    # classifier.to(device)
    deep_lab_v3_plus = DeepLabV3Plus(model, classifier)
    deep_lab_v3_plus.to(device)
    input = torch.rand(1, 3, 32, 32)
    input = input.to(device)
    import torch
    import numpy as np
    torch.set_printoptions(threshold=np.inf)
    output = deep_lab_v3_plus(input)
    print(output.size())
    # print(output)
    print(torch.argmax(output, dim=1).size())
    print(torch.argmax(output, dim=1))
    # # print(output)