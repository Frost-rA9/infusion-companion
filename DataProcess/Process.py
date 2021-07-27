# encoding=GBK
"""DataProcess

    - ����ְ����ǵõ�Ԥ����

    - ˼·���̣�

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
    print_var = False  # ���������Ƿ��ӡ�м���Ϣ

    def __init__(self,
                 svm_path="../Resource/svm/trained/new_bottle_svm.svm",
                 yolo_wight="../Resource/model_data/test_model/yolo/Epoch100-Total_Loss6.6752-Val_Loss11.2832.pth",
                 yolo_anchors="../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../Resource/model_data/infusion_classes.txt",
                 liquid_model_path="../Resource/model_data/test_model/GoogLeNet/loss_7.37_acc_0.875_model.pth",
                 expression_model_path="../Resource/model_data/test_model/FaceCnn/new_face_cnn.pth"
                 ):

        print('{:*^80}'.format("start init all weight...."))
        self.object_locate = ObjectLocate(svm_path, yolo_wight, yolo_anchors, yolo_predict_class)
        self.liquid_level_detect = LiquidDetectCombine(classifier_model_path=liquid_model_path, use_multi_threshold=True)
        self.expression_detect = ExpressionDetectWithFaceCnn(expression_model_path)

        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # ������
        # Һλ��һ���仯�����Ķ�����ÿ��Ӧ�÷��ؽ����ƽ��ֵ, 0��Ϊ�˷�ֹ����
        # �ڿ�ʼ��⼸������0��Ӱ��Ϳ��Ժ�����
        self.liquid_level = [0]  # ����Ǵ�����ָ��õã��Ժ�ķ���
        self.liquid_level_dict = {0: "un_detect"}  # �������ö��ֵķ�����Һλ�ģ�����ʹ����
        for mean in LoadSingleFile.google_expression_dict:
            index = LoadSingleFile.google_expression_dict[mean]
            self.liquid_level_dict[index + 1] = mean
        # ������ֵ�
        self.expression_dict = {0: "un_detect"}
        for mean in LoadSingleFile.expression_dict:
            index = LoadSingleFile.expression_dict[mean]
            self.expression_dict[index + 1] = mean

        print('{:*^80}'.format("weight init finished...."))

    def process_seq(self, img: np.ndarray):
        # 0. ���ݴ�Сͬ��
        img = cv.resize(img, (800, 600))
        # print("process_seq img size", img.shape)

        # 1. ��ȡroi
        loc_list = self.object_locate.get_loc(img)

        if DataProcess.print_var:
            print("roi list��", loc_list)

        bottle_loc, face_loc = loc_list
        bottle_roi, face_roi = [], []

        if bottle_loc:
            for start, end in bottle_loc:
                cv.rectangle(img, start, end, color=self.color_list[0], thickness=2)  # ��ͼ������
                left, top = start
                right, bottom = end
                bottle_roi.append(img[left:right, top:bottom])
        if face_loc:
            for start, end in face_loc:
                cv.rectangle(img, start, end, color=self.color_list[1], thickness=2)
                left, top = start
                right, bottom = end
                face_roi.append(img[left:right, top:bottom])

        # 2. ���ڶ�λ�ɹ������ݽ��д���
        # ���list����������
        bottle_list = [0] * len(self.liquid_level_dict)
        if bottle_roi:
            for roi in bottle_roi:
                if len(roi) != 0:  # ��Щʱ��ü�����
                    level = self.liquid_level_detect.level_predict(roi)
                    level = int(level)  # ��ֹ���Ǹ������ʱ�򱨴�
                    # self.liquid_level.append(level)
                    print("level:", level)
                    bottle_list[level + 1] += 1
        liquid_level = self.liquid_level_dict[bottle_list.index(max(bottle_list))]

        expression_list = [0] * len(self.expression_dict)
        if face_roi:
            for roi in face_roi:
                if len(roi) != 0:
                    expression = self.expression_detect.predict(roi)
                    expression_list[expression + 1] += 1
        # ��Ҫ�ǵ�ǰ�������ӣ���ȡͶƱ����
        # ��������и������������ô����ø�
        expression = self.expression_dict[expression_list.index(max(expression_list))]

        if DataProcess.print_var:
            print("expression list��", expression_list)  # ������

        # 3. �Խ�����з���
        return img, liquid_level, expression
        # return img, sum(self.liquid_level) / len(self.liquid_level), expression  # ���ǲ����õ�
        # return sum(self.liquid_level) / len(self.liquid_level), expression


"""-------------����Ϊ���Դ���----------"""


def get_pic(step: int = 1):
    video = cv.VideoCapture("../Report/lower_and_face.avi")

    ret, frame = video.read()
    while True:
        for i in range(step):  # ����Ϊ���ټ�֡
            ret, frame = video.read()
            if not ret:
                break
        yield frame


if __name__ == '__main__':
    data_process = DataProcess()
    import time
    for frame in get_pic(step=16):
        start = time.time()
        img, level, expression = data_process.process_seq(frame)
        print("*" * 100)
        print("liquid level��", level)
        print("expression: ", expression)
        end = time.time()
        print("rate is:", 1 / (end - start))
        print("*" * 100)
        cv.imshow("loc img��", img)
        cv.waitKey(1)
