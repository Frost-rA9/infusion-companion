# encoding=GBK
"""DataProcess

    - 核心职责就是得到预测结果

    - 思路流程：

"""

import cv2 as cv
import numpy as np
from DataProcess.ExpressionDetect.ExpressionDetect import ExpressionDetect
from DataProcess.LiquidLevelDetect.LiquidLevelDetect import LiquidLevelDetect
from DataProcess.ObjectLoacte.ObjectLocate import ObjectLocate


class DataProcess:
    def __init__(self,
                 svm_path="../Resource/svm/trained/bottle_svm.svm",
                 yolo_wight="../Resource/model_data/test_model/yolo/bottle_and_face.pth",
                 yolo_anchors="../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../Resource/model_data/infusion_classes.txt",
                 liquid_model_path="../Resource/model_data/test_model/DeepLabV3Plus/loss_81.27143794298172_0.9152_.pth",
                 expression_model_path="../Resource/model_data/test_model/GoogLeNet/0.6515_model.pth"
                 ):
        self.object_locate = ObjectLocate(svm_path, yolo_wight, yolo_anchors, yolo_predict_class)
        self.liquid_level_detect = LiquidLevelDetect(liquid_model_path)
        self.expression_detect = ExpressionDetect(expression_model_path)

        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 测试用
        # 液位是一个变化很慢的东西，每次应该返回结果的平均值, 0是为了防止报错
        # 在开始检测几秒后，这个0的影响就可以忽略了
        self.liquid_level = [0]
        # 表情的字典
        self.expression_dict = {
            0: "un_detect",
            1: "anger",  # 生气
            2: "disgust",  # 厌恶
            3: "fear",  # 恐惧
            4: "happy",  # 开心
            5: "sad",  # 伤心
            6: "surprised",  # 惊讶
            7: "normal"  # 中性
        }

    def process_seq(self, img: np.ndarray):
        # 0. 数据大小同步
        img = cv.resize(img, (800, 600))

        # 1. 获取roi
        loc_list = self.object_locate.get_loc(img)
        print("roi list：", loc_list)
        bottle_loc, face_loc = loc_list
        bottle_roi, face_roi = [], []

        if bottle_loc:
            for start, end in bottle_loc:
                cv.rectangle(img, start, end, color=self.color_list[0], thickness=2)  # 绘图测试用
                left, top = start
                right, bottom = end
                bottle_roi.append(img[left:right, top:bottom])
        if face_loc:
            for start, end in face_loc:
                cv.rectangle(img, start, end, color=self.color_list[1], thickness=2)
                left, top = start
                right, bottom = end
                face_roi.append(img[left:right, top:bottom])

        # 2. 对于定位成功的数据进行处理
        if bottle_roi:
            for roi in bottle_roi:
                if len(roi) != 0:  # 有些时候裁剪出空
                    level = self.liquid_level_detect.level_predict(roi)
                    self.liquid_level.append(level)

        expression_list = [0] * len(self.expression_dict)
        if face_roi:
            for roi in face_roi:
                if len(roi) != 0:
                    expression = self.expression_detect.predict(roi)
                    expression_list[expression + 1] += 1
        # 主要是当前环境复杂，采取投票机制
        # 后续如果有更多的人脸，那么必须得改
        expression = self.expression_dict[max(expression_list)]

        print("expression list：", expression_list)  # 测试用

        # 3. 对结果进行返回
        return img, sum(self.liquid_level) / len(self.liquid_level), expression  # 这是测试用的
        # return sum(self.liquid_level) / len(self.liquid_level), expression


"""-------------以下为测试代码----------"""


def get_pic():
    video = cv.VideoCapture(1)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    data_process = DataProcess()
    for frame in get_pic():
        img, level, expression = data_process.process_seq(frame)
        print("liquid level：", level)
        print("expression：", expression)
        cv.imshow("loc img：", img)
        cv.waitKey(1)