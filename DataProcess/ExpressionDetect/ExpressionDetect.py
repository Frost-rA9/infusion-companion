# encoding=GBK
"""ExpressionDetect

        - 表情检测并分类
            - 检测调用了ObjectLocate的内容
            - 分类用了自己的方法(DeepFace除外，它直接全部做完了)
"""

import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
from utils.ImageConvert.ConvertColorSpace import ConvertColorSpace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExpressionDetect:
    def __init__(self,
                 model_path="../../Resource/model_data/test_model/GoogLeNet/0.6515_model.pth"):
        # 注意这个GoogLeNet在CPU上跑，防止显卡内存溢出
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
        return predict_data  # 获取类别交给DataProcess.py


class ExpressionDetectWithFaceCnn:
    def __init__(self,
                 model_path: str = "../../Resource/model_data/face_cnn_loss_5.96_acc_1.0.pth"):
        print("Loading face cnn")
        self.model = torch.load(model_path, map_location=device)
        print("Face Cnn Load finish")

    def predict(self, img: np.ndarray):
        # 1. 通道数变换
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
    img_path = "../../Resource/DataSet/CAER-S/0/1.jpg"
    img = cv.imread(img_path)
    # expression = ExpressionDetect()
    expression = ExpressionDetectWithFaceCnn()
    print("expression:", expression.predict(img))
