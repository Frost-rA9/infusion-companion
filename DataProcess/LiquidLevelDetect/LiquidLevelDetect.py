# encoding=GBK
"""LiquidLevelDetect

    - 液位检测的主逻辑部分

    - 传统方案：
        - 阈值检测

    - 模型方案：
        - DeepLabV3分割

"""
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
import torch.nn as nn

from utils.Caculate.LiquidLeftCal import LiquidLeftCal
# from net.DeepLabV3Plus.DeepLabV3Plus import DeepLabV3Plus


class LiquidLevelDetect:
    def __init__(self,
                 model_path="../../Resource/model_data/test_model/DeepLabV3Plus/loss_81.27143794298172_0.9152_.pth"):
        print("init deepLabV3...")
        self.model = torch.load(model_path)
        print("deepLab load finish..")
        self.liquid_cal = LiquidLeftCal()

    def level_predict(self, img: np.ndarray):
        # 0. resize
        img = cv.resize(img, (400, 300))

        # 1. img->tensor
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)

        # 2. predict
        predict_img = self.model(tensor_img.cuda())

        # 3. predict->numpy
        predict_img = nn.Softmax(dim=1)(predict_img)
        predict_img = torch.argmax(predict_img, dim=1)
        predict_img = predict_img.squeeze(0)
        predict_data = predict_img.data.cpu().numpy()

        # 4. cal
        predict_level = self.liquid_cal.get_cur_liquid(predict_data)
        # img = self.liquid_cal.predict_show(predict_data)
        # return img

        # 5. return predict
        return predict_level


if __name__ == '__main__':
    img_path = "F:/DataSet/bottle/segmentation/dir_json/train/2_json/img.png"
    img = cv.imread(img_path)
    liquid = LiquidLevelDetect()
    data = liquid.level_predict(img)
    print(data)
