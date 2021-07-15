# encoding=GBK
"""DataProcess

    - ����ְ����ǵõ�Ԥ����

    - ˼·���̣�

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

        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # ������
        # Һλ��һ���仯�����Ķ�����ÿ��Ӧ�÷��ؽ����ƽ��ֵ, 0��Ϊ�˷�ֹ����
        # �ڿ�ʼ��⼸������0��Ӱ��Ϳ��Ժ�����
        self.liquid_level = [0]
        # ������ֵ�
        self.expression_dict = {
            0: "un_detect",
            1: "anger",  # ����
            2: "disgust",  # ���
            3: "fear",  # �־�
            4: "happy",  # ����
            5: "sad",  # ����
            6: "surprised",  # ����
            7: "normal"  # ����
        }

    def process_seq(self, img: np.ndarray):
        # 0. ���ݴ�Сͬ��
        img = cv.resize(img, (800, 600))

        # 1. ��ȡroi
        loc_list = self.object_locate.get_loc(img)
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
        if bottle_roi:
            for roi in bottle_roi:
                if len(roi) != 0:  # ��Щʱ��ü�����
                    level = self.liquid_level_detect.level_predict(roi)
                    self.liquid_level.append(level)

        expression_list = [0] * len(self.expression_dict)
        if face_roi:
            for roi in face_roi:
                if len(roi) != 0:
                    expression = self.expression_detect.predict(roi)
                    expression_list[expression + 1] += 1
        # ��Ҫ�ǵ�ǰ�������ӣ���ȡͶƱ����
        # ��������и������������ô����ø�
        expression = self.expression_dict[max(expression_list)]

        print("expression list��", expression_list)  # ������

        # 3. �Խ�����з���
        return img, sum(self.liquid_level) / len(self.liquid_level), expression  # ���ǲ����õ�
        # return sum(self.liquid_level) / len(self.liquid_level), expression


"""-------------����Ϊ���Դ���----------"""


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
        print("liquid level��", level)
        print("expression��", expression)
        cv.imshow("loc img��", img)
        cv.waitKey(1)