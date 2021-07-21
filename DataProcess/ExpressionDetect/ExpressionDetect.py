# encoding=GBK
"""ExpressionDetect

        - �����Ⲣ����
            - ��������ObjectLocate������
            - ���������Լ��ķ���(DeepFace���⣬��ֱ��ȫ��������)
"""

import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
from utils.ImageConvert.ConvertColorSpace import ConvertColorSpace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""ExpressionDetect:
    
    - ʹ��GoogleNetV4���е���������
    
    - Ч�����ѣ����ѷ����������Է�ֹ����
"""


class ExpressionDetect:
    def __init__(self,
                 model_path="../../Resource/model_data/test_model/GoogLeNet/0.6515_model.pth"):
        # ע�����GoogLeNet��CPU���ܣ���ֹ�Կ��ڴ����
        print("init GoogLeNet..")
        self.model = torch.load(model_path)
        print("GoogLeNet init finished")

    def predict(self, img: np.ndarray):
        # 0. resize
        img = cv.resize(img, (400, 400))

        # 1. img->tensor
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)

        # 2. predict
        predict_img = self.model(tensor_img.cuda())

        # 3. predict->numpy
        predict_img = torch.argmax(predict_img, dim=1)
        predict_img = predict_img.squeeze(0)
        predict_data = predict_img.data.cpu().numpy()

        # print(predict_data)

        # 4. get class
        # predict_expression = self.expression_dict[int(predict_data)]

        # 5. return predict
        # return predict_expression
        return predict_data  # ��ȡ��𽻸�DataProcess.py


"""ExpressionDetectWithFaceCnn:
    
    - ����ȵ��������������ʶ�������汾
    
    1. �����������
    2. �Ż���ִ��Ч��
    3. �����׼ȷ��
    
"""


class ExpressionDetectWithFaceCnn:
    def __init__(self,
                 model_path: str = "../../Resource/model_data/1.7163960743040647_0.9864binary_model.pth"):
        print("Loading face cnn")
        self.model = torch.load(model_path, map_location=device)
        print("Face Cnn Load finish")

    def predict(self, img: np.ndarray):
        # 1. ͨ�����任
        if len(img.shape) == 3:
            img = ConvertColorSpace.bgr_and_gray(to_gray=True, img=img)

        # 2. resize
        img = cv.resize(img, (48, 48))

        # 3. to_tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img = img.unsqueeze(0)

        # 4. predict
        output = self.model(img.to(device))
        output = torch.argmax(output, dim=1)

        # 5. data
        data = output.data.cpu().numpy()[0]
        return data


if __name__ == '__main__':
    expression = ExpressionDetectWithFaceCnn()
    from utils.ImageLoaderHelper.VideoHelper import VideoHelper
    for frame in VideoHelper.read_frame_from_cap(0):
        print("expression:", expression.predict(frame))
        cv.imshow("img", frame)
        cv.waitKey(1)
