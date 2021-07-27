# encoding=GBK
"""DataProcess

    - 核心职责就是得到预测结果

"""

import cv2 as cv
import numpy as np
from utils.DataLoader.LoadSingleFile import LoadSingleFile
# from DataProcess.ExpressionDetect.ExpressionDetect import ExpressionDetect
from DataProcess.ExpressionDetect.ExpressionDetect import ExpressionDetectWithFaceCnn
# from DataProcess.LiquidLevelDetect.LiquidLevelDetect import LiquidLevelDetect
from DataProcess.LiquidLevelDetect.LiquidLevelDetect import LiquidDetectCombine
from DataProcess.ObjectLoacte.ObjectLocate import ObjectLocate


class DataProcess:
    print_var = False  # 用来控制是否打印中间信息

    def __init__(self,
                 svm_path="../Resource/svm/trained/new_bottle_svm.svm",
                 yolo_wight="../Resource/model_data/test_model/yolo/Epoch100-Total_Loss6.6752-Val_Loss11.2832.pth",
                 yolo_anchors="../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../Resource/model_data/infusion_classes.txt",
                 liquid_model_path="../Resource/model_data/test_model/GoogLeNet/loss_7.37_acc_0.875_model.pth",
                 expression_model_path="../Resource/model_data/test_model/FaceCnn/new_face_cnn.pth",
                 rectangle_area: float = 50,
                 ):

        print('{:*^80}'.format("start init all weight...."))
        # 1. 模型文件初始化
        self.object_locate = ObjectLocate(svm_path, yolo_wight, yolo_anchors, yolo_predict_class, rectangle_area)
        self.liquid_level_detect = LiquidDetectCombine(classifier_model_path=liquid_model_path,
                                                       use_multi_threshold=True)
                                                       # segmentation_model_path="../Resource/model_data/test_model/DeepLabV3plus/loss_81.27131041884422_0.8332_.pth",
                                                       # use_classifier=False)
        self.expression_detect = ExpressionDetectWithFaceCnn(expression_model_path)

        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 测试用

        # 2. 液位参数初始化
        # 液位是一个变化很慢的东西，每次应该返回结果的平均值, 0是为了防止报错, 这项以后更新
        # 在开始检测几秒后，这个0的影响就可以忽略了
        self.liquid_level = [0]  # 这个是纯语义分割用得，以后的方向

        self.liquid_level_dict = {0: "un_detect"}  # 这是是用二分的方案做液位的，现在使用它
        for mean in LoadSingleFile.google_expression_dict:
            index = LoadSingleFile.google_expression_dict[mean]
            self.liquid_level_dict[index + 1] = mean
        # self.now_liquid_level = [0, [0] * len(self.liquid_level_dict)]  # 多次确认机制,应为变化太大了

        # 3. 表情参数初始化
        # 表情的字典
        self.expression_dict = {0: "un_detect"}
        for mean in LoadSingleFile.expression_dict:
            index = LoadSingleFile.expression_dict[mean]
            self.expression_dict[index + 1] = mean

        print('{:*^80}'.format("weight init finished...."))

    def process_seq(self, img: np.ndarray):
        # 0. 数据大小同步
        img = cv.resize(img, (800, 600))

        # 1. 获取roi
        loc_list = self.object_locate.get_loc(img)

        if DataProcess.print_var:
            print("roi list：", loc_list)

        bottle_loc, face_loc = loc_list
        bottle_roi, face_roi = [], []  # 先声明是后续需要这个变量否则报错
        # 不为空的时候截获相关的图像区域
        if bottle_loc:
            bottle_roi = self.get_des_roi(img, bottle_loc, color_index=0)
        if face_loc:
            face_roi = self.get_des_roi(img, face_loc, color_index=1)

        # 2. 对于定位成功的数据进行处理
        bottle_list = [0] * len(self.liquid_level_dict)  # 在定位精度不行的时候使用比如出现了好几个
        if bottle_roi:
            for roi in bottle_roi:
                if len(roi) != 0:  # 有些时候裁剪出空
                    level = self.liquid_level_detect.level_predict(roi)
                    level = int(level)  # 防止忘记改网络的时候报错
                    # self.liquid_level.append(level)
                    if DataProcess.print_var:
                        print("level:", level)
                    bottle_list[level + 1] += 1
        liquid_level = self.liquid_level_dict[bottle_list.index(max(bottle_list))]

        expression_list = [0] * len(self.expression_dict)
        if face_roi:
            for roi in face_roi:
                if len(roi) != 0:
                    expression = self.expression_detect.predict(roi)
                    expression_list[expression + 1] += 1
        # 主要是当前环境复杂，采取投票机制
        # 后续如果有更多的人脸，那么必须得改
        expression = self.expression_dict[expression_list.index(max(expression_list))]

        if DataProcess.print_var:
            print("expression list：", expression_list)  # 测试用

        # 3. 对结果进行返回
        return img, liquid_level, expression
        # return img, sum(self.liquid_level) / len(self.liquid_level), expression  # 这是测试用的
        # return sum(self.liquid_level) / len(self.liquid_level), expression

    def get_des_roi(self, img:np.ndarray, loc_list: list, color_index: int):
        roi_list = []
        for start, end in loc_list:
            cv.rectangle(img, start, end, color=self.color_list[color_index], thickness=2)
            left, top = start
            right, bottom = end
            roi_list.append(img[top: bottom, left: right])
        return roi_list


"""-------------以下为测试代码----------"""


def get_pic(step: int = 1):
    video = cv.VideoCapture(0)

    ret, frame = video.read()
    while True:
        for i in range(step):  # 单纯为了少几帧
            ret, frame = video.read()
            if not ret:
                break
        yield frame


if __name__ == '__main__':
    data_process = DataProcess()
    import time
    for frame in get_pic(step=8):
        if frame is None:
            break
        start = time.time()
        img, level, expression = data_process.process_seq(frame)
        print("*" * 100)
        print("liquid level：", level)
        print("expression: ", expression)
        end = time.time()
        print("rate is:", 1 / (end - start))
        print("*" * 100)
        cv.imshow("loc img", img)
        cv.waitKey(1)
