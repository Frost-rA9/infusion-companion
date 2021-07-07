# encoding=GBK
"""DarkNet53

    - 封装创建模型的参数
"""
from net.BoneNet.DarkNet.DarkNet import DarkNet
import torch


class DarkNet53:
    @staticmethod
    def get_model(block: list = [1, 2, 8, 8, 4],
                  pretrained: str = None):
        model = DarkNet(block)
        if pretrained:
            if isinstance(pretrained, str):
                model.load_state_dict(torch.load(pretrained))
            else:
                raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
        return model
