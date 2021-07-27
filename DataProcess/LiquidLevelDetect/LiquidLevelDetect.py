# encoding=GBK
"""LiquidLevelDetect

    - Һλ�������߼�����

    - ��ͳ������
        - ��ֵ���

    - ģ�ͷ�����
        - DeepLabV3�ָ�

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

    - ʹ�ô�����ָ�ķ��������ص�����и�
    - Ȼ��ͨ��ͳ����ˮ������ռ��ˮ+��ˮ�ı�ֵ�ﵽ����Ч��
    
    - ʵ�ʲ��Թ����У����Ч���ϲ������ϵ������汾�ļ�ⷽ����
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
    
    - combine��3���㷨(����ָ�ģ�Ϳ�ʵ�����ʹ��)
    1. GoogLeNet�Ķ�����ģ��
    2. opencv�Ĵ�ͳԤ��ģ��
    3. ��������ָ������ֵ����
    
    - ����2��3���ֿ�����Ϊ�Ժ�׼Ԥ��ʱʹ�ã��������ڵ�׼ȷ�ʲ�զ��
    - ���Խ���ѡ���Կ���
    - ע������ľ�ȷҺλ������������Χ�ģ���<0.4��Ϊ��lower������Ϊupper
    
    - ���ڵ�׼ȷ�ȶԱ����£����ֵ�׼ȷ�ʣ�
        - ��������Ӧ���������ָ�ģ�͵ľ�ȷ�ȣ�����GoogLeNet���з���
        - ����ʹ�ô�GoogLeNet���з���
        
        - ����׼ȷ�ʽ����ο���ʵ��ʹ��Ӧ�ô����
        
        - ׼ȷ�ʣ�Ԥ��Ϊ�������͵�ͼ��ռ�����������ݼ��ı���
        - upper: 354�� lower: 512��
    
        - �����������£�
            
        . ������(GoogLeNetV4):
            - loss_29.43_acc_0.8125_model.pth:
                - �Դ�ռ��1.1G
                - upper����׼ȷ�ʣ�0.9180790960451978
                - lower����׼ȷ�ʣ�0.984375
            - loss_7.37_acc_0.875_model.pth:
                - �Դ�ռ��1.1G
                - upper����׼ȷ�ʣ�1.0
                - lower����׼ȷ�ʣ�0.99609375
                - ʵ��ʹ�õ�ʹ��Ԥ��ֵ�෴������̫��ѵ������û�������������
        . opencv:
            - �Դ�ռ��0
            - upper����׼ȷ�ʣ�0.576271186440678
            - lower����׼ȷ�ʣ�0.619140625
        . ����ָ�(DeepLabV3Plus):
            - �Դ�ռ��: 1.9G
            - upper����׼ȷ�ʣ�0.7966101694915254
            - lower����׼ȷ�ʣ�0.66015625
            
        . ������ + opencv:
            - �Դ�ռ��1.1G 
            - Ȩ��1:1
                - upper����׼ȷ�ʣ�0.9548022598870056
                - lower����׼ȷ�ʣ�0.609375
            - Ȩ��1��0.5
                - upper����׼ȷ�ʣ�0.9180790960451978
                - lower����׼ȷ�ʣ�0.984375
        . ������ + ����ָ�
            - �Դ�ռ��2.4G 
            - Ȩ��1:1
                - upper����׼ȷ�ʣ�0.9717514124293786
                - lower����׼ȷ�ʣ�0.716796875
        . ����ָ� + opencv
            - �Դ�ռ��2.0G
            - Ȩ��1:1
                - upper����׼ȷ�ʣ�0.8333333333333334
                - lower����׼ȷ�ʣ�0.46484375
        
        . ������ + ����ָ� + opencv
            - �Դ�ռ��2.4G
            - Ȩ��1:1:1
                - upper����׼ȷ�ʣ�0.7937853107344632
                - lower����׼ȷ�ʣ�0.880859375
            - Ȩ��1.5:1:0.5
                - upper����׼ȷ�ʣ�0.8220338983050848
                - lower����׼ȷ�ʣ�0.7265625
        
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
        # 1. ��ʼ���ȸ������
        if use_classifier:
            print("classifier model init ..")
            self.classifier = torch.load(classifier_model_path, map_location=device)
            print("classifier load finish..")
        else:
            self.classifier = None

        # 2. ���������ʼ������ָ�ģ��
        if segmentation_model_path:
            print("Segmentation model init ..")
            self.segmentation = torch.load(segmentation_model_path, map_location=device)
            print("Segmentation load finish ..")
            self.liquid_cal = LiquidLeftCal()
        else:
            self.segmentation = None

        # 3. �����������ͳԤ��
        if use_multi_threshold:
            self.multi_threshold_detect = ThresholdDetect2()
        else:
            self.multi_threshold_detect = None

        # 4. ����ֵ��Ժ�Ӧ�øĳ����
        self.binary_dict = {
            0: "upper",
            1: "lower"
        }
        self.weight_list = weight_list

    def level_predict(self, img: np.ndarray):
        classifier_weight, segmentation_weight, opencv_weight = self.weight_list
        vote_list = [0, 0]  # 0: upper, 1: lower

        # 0. use_multi_threshold
        # opencv��Ԥ����������ü����ԭͼ��������resize
        if self.multi_threshold_detect:
            data = self.multi_threshold_detect.cal_seq(img)
            if data < 0.4:  # ��������ȫ��������Ԥ��lower��ƽ��ֵ
                index = 1
            else:
                index = 0
            vote_list[index] += opencv_weight  # Ȩ��

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
            class_result = self.classifier(tensor_img.to(device))  # ��������������״̬ 0:upper, 1:lower
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
                predict_level = self.liquid_cal.get_cur_liquid(predict_data)  # �ָ�������Һλ��״̬
                if predict_level < 0.4:  # ����ָ�����������
                    index = 1
                else:
                    index = 0
                vote_list[index] += segmentation_weight

                if LiquidDetectCombine.print_var:
                    print("Segmentation predict index:", index)

            except Exception as e:
                if ThresholdDetect2.print_var:
                    print(e, "�������Ѿ�������")

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
