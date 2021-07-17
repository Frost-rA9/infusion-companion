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
                 yolo_wight="../../Resource/model_data/test_model/yolo/Epoch100-Total_Loss7.1096-Val_Loss12.4228.pth",
                 yolo_anchors="../../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../../Resource/model_data/infusion_classes.txt",
                 ):
        self.bottle_locate_svm = LocateRoI(svm_path)
        self.yolo_locate = YoLoLocate(yolo_wight, yolo_anchors, yolo_predict_class)
        self.cascade = Cascade()

    def get_loc(self, img: np.ndarray, focus_on_model: bool = True):
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

        :param focus_on_model: ����dlib�Ķ�λЧ�����ǽϲ����yolov3����һƱ���Ȩ
        :param img: np.ndarray��ʽ
        :return: loc_list
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

                has_bottle = False  # yolo��һƱ���Ȩ

                for data_list in loc_data:
                    index = classes_list.index(data_list[0])
                    if index == 0:
                        has_bottle = True
                    # ( (left, top), (right, end) )
                    loc_list[index].append(((data_list[2], data_list[3]), (data_list[4], data_list[5])))

                if focus_on_model and not has_bottle:
                    loc_list[0] = False  # ��dlib�Ķ�λ���
                    print("dlib was rejected by yolov3")

        # ���������ĺϲ�
        loc_list = self.roi_filter(loc_list)
        return self.reliable_detect(loc_list)  # ת��Ϊ���Ž��

    def roi_filter(self, loc_list: list):
        bottle_list, face_list = loc_list
        bottle_list = self.get_max_face_roi(bottle_list)
        face_list = self.get_max_face_roi(face_list)
        loc_list[0] = bottle_list
        loc_list[1] = face_list
        return loc_list

    def get_max_face_roi(self, face_list: list):
        max_width, max_height = None, None
        new_face_list = []

        if face_list is False:
            return False

        for data in face_list:
            if not data:
                pass
            (left, top), (right, end) = data
            # 1. û�о�ֱ�Ӹ�ֵ
            # 2. �еĻ����㽻�������������ϴ���ôѡȡ�߽�ϴ��һ����Ϊ�߽�
            if not max_width:
                max_width = (left, right)
            else:
                last_left, last_right = max_width
                # 1. �ҳ�������
                max_left = max([last_left, left])
                min_right = min([last_right, right])

                # 2. �ж������Ƿ��޹�
                if min_right <= max_left:
                    new_face_list.append(( (left, top), (right, end) ))
                    pass
                else:
                    horizontal_cross = min_right - max_left
                    horizontal_area = max(last_right, right) - min(last_left, left)
                    horizontal_roi = horizontal_cross / horizontal_area

            if not max_height:
                max_height = (top, end)
            else:
                last_top, last_end = max_height
                # 1. �ҳ�������
                max_top = max(last_top, top)
                min_end = min(last_end, end)

                # 2. �ж��Ƿ��޹�
                if max_top >= min_end:
                    new_face_list.append(((left, top), (right, end)))
                    pass
                else:
                    vertical_cross = min_end - max_top
                    vertical_area = max(last_end, end) - min(last_top, top)
                    vertical_roi = vertical_cross / vertical_area

            try:
                roi = horizontal_roi * vertical_roi
                print("Cross area roi is: ", roi)
                if roi > 0.6:  # ˵���ص��Ⱥܸߣ�ȡ��Χ���
                    last_left, last_right = max_width
                    max_width = (min(last_left, left), max(last_right, right))
                    last_top, last_end = max_height
                    max_height = (min(last_top, top), max(last_end, end))
            except:
                pass

        # ȫ��ѭ����Ϻ󷵻ؽ��
        if not max_width or not max_height:
            return False
        else:
            left, right = max_width
            top, end = max_height
            # ( (left, top), (right, end) )
            new_face_list.append(((left, top), (right, end)))
            return new_face_list

    def reliable_detect(self, loc_list: list):
        """ʵ��ʹ�ù����г����˸����� ������Ҫ����"""
        # [[((435, -159), (733, 585))], [((546, 305), (667, 426))]]
        bottle_loc, face_loc = loc_list
        new_bottle_loc = []
        if bottle_loc:
            for index in range(len(bottle_loc)):
                start, end = bottle_loc[index]
                left, top = start
                right, end = end
                temp_list = list(map(abs, [left, top, right, end]))
                if temp_list[0] > temp_list[2]:
                    pass
                if temp_list[1] > temp_list[3]:
                    pass

                new_bottle_loc.append(((temp_list[0], temp_list[1]), (temp_list[2], temp_list[3])))
        if len(new_bottle_loc) == 0:
            bottle_loc = False

        new_face_loc = []
        if face_loc:
            for index in range(len(face_loc)):
                start, end = face_loc[index]
                left, top = start
                right, end = end
                temp_list = list(map(abs, [left, top, right, end]))
                if temp_list[0] > temp_list[2]:
                    pass
                if temp_list[1] > temp_list[3]:
                    pass

                new_face_loc.append(((temp_list[0], temp_list[1]), (temp_list[2], temp_list[3])))
        if len(new_face_loc) == 0:
            face_loc = False

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
    video = cv.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    _object = ObjectLocate()

    # error_list = [False, [((340, 166), (510, 406)), ((322, 173), (527, 378))]]
    # error_list = [False, False]
    # bottle_list, face_list = error_list
    # print(_object.get_max_face_roi(face_list))
    # error_list = [False, [((546, 305), (667, 426))]]
    # error_list = [[((321, -174), (570, 446))], False]
    # print(_object.reliable_detect(error_list))

    import time
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for pic in get_pic():
        start_time = time.time()

        pic = cv.resize(pic, (800, 600))
        loc_list = _object.get_loc(pic)
        bottle_loc, face_loc = loc_list
        print(loc_list)
        if bottle_loc:
            for start, end in bottle_loc:
                cv.rectangle(pic, start, end, color=color_list[0], thickness=2)
        if face_loc:
            for start, end in face_loc:
                cv.rectangle(pic, start, end, color=color_list[1], thickness=2)

        cv.imshow("pic", pic)
        cv.waitKey(1)

        end_time = time.time()
        speed_time = end_time - start_time
        print("-------------image rate is:{}----------".format(str(1 / speed_time)))
