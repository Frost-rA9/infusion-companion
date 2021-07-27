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


"""LiquidDetectCombine:
    
    - combine了3种算法(语义分割模型看实际情况使用)
    1. GoogLeNet的二分类模型
    2. opencv的传统预测模型
    3. 基于语义分割的像素值运算
    
    - 其中2，3两种可以作为以后精准预测时使用，但是现在的准确率不咋地
    - 可以进行选择性开启
    - 注意这里的精确液位检测是添加了误差范围的，即<0.4认为是lower，否则为upper
    
    - 现在的准确度对比如下（二分的准确率）
        - 测试下来应该提高语义分割模型的精确度，辅助GoogLeNet进行分类
        - 或者使用纯GoogLeNet进行分类
        
        - 另外准确率仅供参考，实际使用应该打八折
        
        - 准确率：预测为这种类型的图像占这种类型数据集的比例
        - upper: 354张 lower: 512张
    
        - 测量数据如下：
            
        . 分类器(GoogLeNetV4):
            - loss_29.43_acc_0.8125_model.pth:
                - 显存占用1.1G
                - upper数据准确率：0.9180790960451978
                - lower数据准确率：0.984375
            - loss_7.37_acc_0.875_model.pth:
                - 显存占用1.1G
                - upper数据准确率：1.0
                - lower数据准确率：0.99609375
                - 实际使用的使用预测值相反（干扰太大？训练数据没出现这种情况）
        . opencv:
            - 显存占用0
            - upper数据准确率：0.576271186440678
            - lower数据准确率：0.619140625
        . 语义分割(DeepLabV3Plus):
            - 显存占用: 1.9G
            - upper数据准确率：0.7966101694915254
            - lower数据准确率：0.66015625
            
        . 分类器 + opencv:
            - 显存占用1.1G 
            - 权重1:1
                - upper数据准确率：0.9548022598870056
                - lower数据准确率：0.609375
            - 权重1：0.5
                - upper数据准确率：0.9180790960451978
                - lower数据准确率：0.984375
        . 分类器 + 语义分割
            - 显存占用2.4G 
            - 权重1:1
                - upper数据准确率：0.9717514124293786
                - lower数据准确率：0.716796875
        . 语义分割 + opencv
            - 显存占用2.0G
            - 权重1:1
                - upper数据准确率：0.8333333333333334
                - lower数据准确率：0.46484375
        
        . 分类器 + 语义分割 + opencv
            - 显存占用2.4G
            - 权重1:1:1
                - upper数据准确率：0.7937853107344632
                - lower数据准确率：0.880859375
            - 权重1.5:1:0.5
                - upper数据准确率：0.8220338983050848
                - lower数据准确率：0.7265625
        
"""

from DataProcess.LiquidLevelDetect.DynamicThresholdDetection import ThresholdDetect2


class LiquidDetectCombine:
    print_var = False

    def __init__(self,
                 classifier_model_path: str = "../../Resource/model_data/test_model/GoogLeNet/loss_7.37_acc_0.875_model.pth",
                 segmentation_model_path: str = None,
                 use_multi_threshold: bool = False,
                 use_classifier: bool = True,
                 weight_list: list = [1, 1, 1],
                 ):
        # 1. 初始化谷歌分类器
        if use_classifier:
            print("classifier model init ..")
            self.classifier = torch.load(classifier_model_path, map_location=device)
            print("classifier load finish..")
        else:
            self.classifier = None

        # 2. 按照情况初始化语义分割模型
        if segmentation_model_path:
            print("Segmentation model init ..")
            self.segmentation = torch.load(segmentation_model_path, map_location=device)
            print("Segmentation load finish ..")
            self.liquid_cal = LiquidLeftCal()
        else:
            self.segmentation = None

        # 3. 看情况开启传统预测
        if use_multi_threshold:
            self.multi_threshold_detect = ThresholdDetect2()
        else:
            self.multi_threshold_detect = None

        # 4. 这个字典以后应该改成相对
        self.binary_dict = {
            0: "upper",
            1: "lower"
        }
        self.weight_list = weight_list

    def level_predict(self, img: np.ndarray):
        classifier_weight, segmentation_weight, opencv_weight = self.weight_list
        vote_list = [0, 0]  # 0: upper, 1: lower

        # 0. use_multi_threshold
        # opencv的预测最好贴近裁剪框的原图，而不是resize
        if self.multi_threshold_detect:
            data = self.multi_threshold_detect.cal_seq(img)
            if data < 0.4:  # 这是现在全局跑下来预测lower的平均值
                index = 1
            else:
                index = 0
            vote_list[index] += opencv_weight  # 权重

            if LiquidDetectCombine.print_var:
                print("opencv predict index", index)

        # 1. resize
        img = cv.resize(img, (400, 400))

        # 2. img->tensor
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)

        # 3. predict
        if self.classifier:
            class_result = self.classifier(tensor_img.to(device))  # 二分类结果的两种状态 0:upper, 1:lower
            class_result = class_result.data.cpu().numpy()[0]
            class_result = list(class_result)
            index = class_result.index(max(class_result))
            vote_list[index] += classifier_weight

            if LiquidDetectCombine.print_var:
                print("Classifier predict index: ", index)

        if self.segmentation:
            predict_img = self.segmentation(tensor_img.to(device))
            predict_img = torch.argmax(predict_img, dim=1)
            predict_img = predict_img.squeeze(0)
            predict_data = predict_img.data.cpu().numpy()
            try:
                predict_level = self.liquid_cal.get_cur_liquid(predict_data)  # 分割结果出来液位的状态
                if predict_level < 0.4:  # 语义分割加上误差的情况
                    index = 1
                else:
                    index = 0
                vote_list[index] += segmentation_weight

                if LiquidDetectCombine.print_var:
                    print("Segmentation predict index:", index)

            except Exception as e:
                if ThresholdDetect2.print_var:
                    print(e, "此项结果已经被忽略")

        # 4. draw conclusion
        index = vote_list.index(max(vote_list))

        if LiquidDetectCombine.print_var:
            print("liquidDetectCombine predict index: ", index)

        return index
        # return self.binary_dict[index]


if __name__ == '__main__':
    import os
    from PIL import Image
    from utils.ImageLoaderHelper.VideoHelper import VideoHelper

    segmentation_path = "../../Resource/model_data/test_model/DeepLabV3plus/loss_81.27131041884422_0.8332_.pth"
    liquid_combine = LiquidDetectCombine(
        # use_multi_threshold=True,
        # segmentation_model_path=segmentation_path,
        use_classifier=True,
    )
    binary_dict = {
        0: "upper",
        1: "lower"
    }
    LiquidDetectCombine.print_var = True
    dir = "F:/temp/Classifer/train/upper"
    all, lower = 0, 0
    for file in os.listdir(dir):
        file = dir + "/" + file
        img = cv.imread(file)
        result = liquid_combine.level_predict(img)
        print(result)
        if binary_dict[result] == 'lower':
            lower += 1
        all += 1
    print(lower / all)

    ### for LiquidLevelDetect test
    # dir = "F:/temp/Classifer/train/upper"
    # liquid = LiquidLevelDetect()
    # liquid_cal = LiquidLeftCal()
    # lower, all = 0, 0
    # for d in os.listdir(dir):
    #     img_path = dir + '/' + d
    #     img = cv.imread(img_path)
    #     data1 = liquid.level_predict(img)
    #     print(data1)
    #
    #     if data1 < 0.4:
    #         lower += 1
    #     all += 1
    # print("lower rate:", lower / all)
