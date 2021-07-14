# encoding=GBK
"""ObjectLocate

    - ����ObjectLocateĿ¼����������
    - ���Ⱪ¶һ����ȡ������λ��ƿ�Ӷ�λ�ĺ���(get_loc)
        - ���������������
"""
import numpy as np
import cv2 as cv
from utils.LocateObject.dlibLocate import LocateRoI  # ��λƿ��
from utils.LocateObject.Cascade import Cascade  # ��λ����
from utils.LocateObject.YoLoLocate import YoLoLocate  # ��λ����


class ObjectLocate:
    def __init__(self,
                 svm_path="../../Resource/svm/trained/bottle_svm.svm",
                 yolo_wight="../../Resource/model_data/test_model/yolo/Epoch12-Total_Loss11.4916-Val_Loss8.8832.pth",
                 yolo_anchors="../../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../../Resource/model_data/infusion_classes.txt",
                 ):
        self.bottle_locate_svm = LocateRoI(svm_path)
        self.yolo_locate = YoLoLocate(yolo_wight, yolo_anchors, yolo_predict_class)
        self.cascade = Cascade()

    def get_loc(self, img: np.ndarray):
        """

        ��ȡ��λ�Ĳ������£�
            1. ���Ƚ���get_bottle_loc
                - ��Ҫ�ж��Ƿ�Ϊ��
                - �հ�loc_list[0] = False
            2. ����get_face_loc
                - ͬ���ж��Ƿ�Ϊ��
                - �հ�loc_list[1] = False
            3. ����пյ�������֣�����YoLo���м��
                - ������Ӧλ�ý������
            4. ����loc_list�п��ܻ����ֵ��False�����
                - ��Ҫ�Լ������

        :param img:
        :return:
        """
        loc_list = [True, True]  # ��ʾ��û�ҵ�

        bottle_loc = self.get_bottle_loc(img)
        # bottle_loc = None
        if not bottle_loc:
            loc_list[0] = False  # û�ҵ��趨��ǣ�����yolo
        else:
            loc_list[0] = bottle_loc  # �ҵ�����һ���б�

        face_loc = self.get_face_loc(img)
        # face_loc = None
        if not face_loc:
            loc_list[1] = False
        else:
            loc_list[1] = face_loc

        # print(False in loc_list)

        if False in loc_list:  # ʵ���Ͼ��Ǽ��ٺ�̨���ݴ��������ʱ��
            loc_data = self.get_all_loc(img)
            if not loc_data:
                return self.reliable_detect(loc_list)  # yolo����ⲻ��ֱ�ӷ���
            else:
                classes_list = ['bottle', 'face']
                for i in range(len(loc_list)):
                    if not loc_list[i]:
                        loc_list[i] = []

                for data_list in loc_data:
                    index = classes_list.index(data_list[0])
                    # ( (left, top), (right, end) )
                    loc_list[index].append(( (data_list[2], data_list[3]), (data_list[4], data_list[5]) ))
        return self.reliable_detect(loc_list)

    def reliable_detect(self, loc_list: list):
        """ʵ��ʹ�ù����г����˸����� ������Ҫ����"""
        # [[((435, -159), (733, 585))], [((546, 305), (667, 426))]]
        bottle_loc, face_loc = loc_list
        if bottle_loc:
            for index in range(len(bottle_loc)):
                start, end = bottle_loc[index]
                left, top = start
                right, end = end
                temp_list = list(map(abs, [left, top, right, end]))
                # if temp_list[0] > temp_list[2]:
                #     pass
                # if temp_list[1] > temp_list[3]:
                #     pass

                bottle_loc[index] = (temp_list[0], temp_list[1]), (temp_list[2], temp_list[3])

        if face_loc:
            for index in range(len(face_loc)):
                start, end = face_loc[index]
                left, top = start
                right, end = end
                temp_list = list(map(abs, [left, top, right, end]))
                # if temp_list[0] > temp_list[2]:
                #     pass
                # if temp_list[1] > temp_list[3]:
                #     pass

                face_loc[index] = (temp_list[0], temp_list[1]), (temp_list[2], temp_list[3])

        loc_list[0] = bottle_loc
        loc_list[1] = face_loc
        return loc_list

    def get_bottle_loc(self, img: np.ndarray):
        """

        ͨ��Svm�ķ����õ�ƿ�ӵ�λ��
        Ȼ����ݷ���ֵ��ȷ���Ƿ����yolo

        :param img:
        :return:
        """
        bottle_loc = self.bottle_locate_svm.predict(img)  # ���ض�λ����
        # bottle_loc = self.bottle_locate_svm.predict_show(img)  # ��������ֱ�ӷ���ͼ��
        return bottle_loc

    def get_face_loc(self, img: np.ndarray):
        face_loc = self.cascade.detect_face(img)
        # img = self.cascade.plot_rect(img)
        return face_loc

    def get_all_loc(self,
                    img: np.ndarray):
        predict_list = self.yolo_locate.predict(img)
        print("YoLo predict", predict_list)
        # predict_list = np.array(self.yolo_locate.draw(img, filter=None, font_path="../../Resource/model_data/simhei.ttf"))
        if predict_list:
            return predict_list[4]  # predicted_class, label, top, left, bottom, right
        else:
            return None


"""______________���Դ���___________________"""


def get_pic():
    video = cv.VideoCapture(1)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    _object = ObjectLocate()

    # error_list = [[((435, -159), (733, 585))], [((546, 305), (667, 426))]]
    # error_list = [False, [((546, 305), (667, 426))]]
    error_list = [[((321, -174), (570, 446))], False]
    print(_object.reliable_detect(error_list))

    # color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # for pic in get_pic():
    #     pic = cv.resize(pic, (800, 600))
    #     loc_list = _object.get_loc(pic)
    #     bottle_loc, face_loc = loc_list
    #     print(loc_list)
    #     if bottle_loc:
    #         for start, end in bottle_loc:
    #             cv.rectangle(pic, start, end, color=color_list[0], thickness=2)
    #     if face_loc:
    #         for start, end in face_loc:
    #             cv.rectangle(pic, start, end, color=color_list[1], thickness=2)
    #
    #     cv.imshow("pic", pic)
    #     cv.waitKey(1)

        # cv.imshow("img", img)
        # cv.waitKey(1)
