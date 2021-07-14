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


class ExpressionDetect:
    def __init__(self,
                 model_path="../../Resource/model_data/test_model/GoogLeNet/0.6515_model.pth"):
        print("init GoogLeNet..")
        self.model = torch.load(model_path)
        print("GoogLeNet init finished")
        # self.expression_dict = {
        #     0: "anger",  # 生气
        #     1: "disgust",  # 厌恶
        #     2: "fear",  # 恐惧
        #     3: "happy",  # 开心
        #     4: "sad",  # 伤心
        #     5: "surprised",  # 惊讶
        #     6: "normal"  # 中性
        # }

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

if __name__ == '__main__':
    img_path = "../../Resource/DataSet/CAER-S/0/1.jpg"
    img = cv.imread(img_path)
    expression = ExpressionDetect()
    print("expression:", expression.predict(img))

