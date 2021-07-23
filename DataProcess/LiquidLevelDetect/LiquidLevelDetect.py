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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""LiquidLevelDetect

    - 使用纯语义分割的方案对像素点进行切割
    - 然后通过统计有水的像素占有水+无水的比值达到计算效果
    
    - 实际测试过程中，拟合效果较差，因此整合到后续版本的检测方案中
"""


class LiquidLevelDetect:
    def __init__(self,
                 model_path="../../Resource/model_data/test_model/DeepLabV3Plus/loss_81.27131041884422_0.8332_.pth"):
        print("init deepLabV3...")
        self.model = torch.load(model_path, map_location=device)
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
        predict_img = self.model(tensor_img.to(device))

        # 3. predict->numpy
        predict_img = nn.Softmax(dim=1)(predict_img)
        predict_img = torch.argmax(predict_img, dim=1)
        predict_img = predict_img.squeeze(0)
        predict_data = predict_img.data.cpu().numpy()

        # 4. cal
        predict_level = self.liquid_cal.get_cur_liquid(predict_data)
        # img = self.liquid_cal.predict_show(predict_data)
        # cv.imshow("img", img)
        # cv.waitKey(0)

        # 5. return predict
        return predict_level


class LiquidDetectCombine:
    print_var = False

    def __init__(self,
                 classifier_model_path: str = "../../Resource/model_data/test_model/GoogLeNet/loss_63.55_acc_0.8928_liquid_binary.pth",
                 segmentation_model_path: str = None,
                 ):
        print("classifier model init ..")
        self.classifier = torch.load(classifier_model_path, map_location=device)
        print("classifier load finish..")

        if segmentation_model_path:
            print("Segmentation model init ..")
            self.segmentation = torch.load(segmentation_model_path, map_location=device)
            print("Segmentation load finish ..")
            self.liquid_cal = LiquidLeftCal()
        else:
            self.segmentation = None

        self.binary_dict = {
            0: "upper",
            1: "lower"
        }

    def level_predict(self, img: np.ndarray):
        # 0. resize
        img = cv.resize(img, (400, 400))

        # 1. img->tensor
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)

        # 2. predict
        class_result = self.classifier(tensor_img.to(device))  # 二分类结果的两种状态 0:upper, 1:lower
        if self.segmentation:
            predict_img = self.classifier(tensor_img.to(device))
            predict_img = torch.argmax(predict_img, dim=1)
            predict_img = predict_img.squeeze(0)
            predict_data = predict_img.data.cpu().numpy()
            predict_level = self.liquid_cal.get_cur_liquid(predict_data)  # 分割结果出来液位的状态
        else:
            predict_level = 0

        if LiquidDetectCombine.print_var:
            print(class_result)

        # 3. weight cal
        class_result = class_result.data.cpu().numpy()[0]
        if predict_level == 0:
            predict_level = np.array([0, 0])  # 0: upper， 1：lower
        elif predict_level > 0.3:
            predict_level = np.array([predict_level, 1-predict_level])
        else:
            predict_level = np.array([1-predict_level, predict_level])
        result = 0.2 * predict_level + 0.8 * class_result

        # 4. draw conclusion
        result = list(result)
        index = result.index(max(result))

        if LiquidDetectCombine.print_var:
            print("liquidDetectCombine predict index: ", index)

        return self.binary_dict[index]


if __name__ == '__main__':
    import os
    from PIL import Image
    from utils.ImageLoaderHelper.VideoHelper import VideoHelper
    liquid_combine = LiquidDetectCombine()
    LiquidDetectCombine.print_var = True
    dir = "F:/temp/Classifer/train/upper"
    all, lower = 0, 0
    for file in os.listdir(dir):
        file = dir + "/" + file
        img = cv.imread(file)
        result = liquid_combine.level_predict(img)
        if result == 'lower':
            lower += 1
        all += 1
    print(lower / all)


    ### for LiquidLevelDetect test
    # file_path = "F:/DataSet/bottle/segmentation/dir_json/train/"
    # liquid = LiquidLevelDetect()
    # liquid_cal = LiquidLeftCal()
    # total = []
    # for d in os.listdir(file_path):
    #     temp = file_path + d + "/"
    #     img_path = temp + "img.png"
    #     img = cv.imread(img_path)
    #     data1 = liquid.level_predict(img)
    #
    #     img_path = temp + "label.png"
    #     i = Image.open(img_path)
    #     i = np.array(i)
    #     data2 = liquid_cal.get_cur_liquid(i)
    #     print("*" * 20)
    #     print("data1", data1, "data2", data2)
    #     differ = abs(data1 - data2) / data2 * 100
    #     print("differ rate is:", differ)
    #     if differ != np.inf:
    #         total.append(differ)
    #     print("*" * 20)
    # print(sum(total) / len(total))
