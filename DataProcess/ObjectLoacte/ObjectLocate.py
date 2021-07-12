# encoding=GBK
"""ObjectLocate

    - 整合ObjectLocate目录中其他的类
    - 对外暴露一个获取人脸定位和瓶子定位的函数(get_loc)
        - 返回坐标后续工作
"""
import numpy as np
import cv2 as cv
from utils.LocateObject.dlibLocate import LocateRoI  # 定位瓶子
from utils.LocateObject.Cascade import Cascade  # 定位人脸
from utils.LocateObject.YoLoLocate import YoLoLocate  # 定位万物


class ObjectLocate:
    def __init__(self,
                 svm_path="../../Resource/svm/trained/bottle_svm.svm",
                 yolo_wight="../../Resource/model_data/test_model/yolo/Epoch12-Total_Loss11.4916-Val_Loss8.8832.pth",
                 yolo_anchors="../../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../../Resource/model_data/infusion_classes.txt"):
        self.bottle_locate_svm = LocateRoI(svm_path)
        self.yolo_locate = YoLoLocate(yolo_wight, yolo_anchors, yolo_predict_class)
        self.cascade = Cascade()

    def get_loc(self, img: np.ndarray):
        """

        获取定位的步骤如下：
            1. 首先进行get_bottle_loc
                - 需要判断是否为空
                - 空把loc_list[0] = False
            2. 进行get_face_loc
                - 同样判断是否为空
                - 空把loc_list[1] = False
            3. 如果有空的情况出现，调用YoLo进行检测
                - 并对相应位置进行填充
            4. 所以loc_list中可能会出现值是False的情况
                - 需要自己做甄别

        :param img:
        :return:
        """
        loc_list = [True, True]  # 表示找没找到

        bottle_loc = self.get_bottle_loc(img)
        if not bottle_loc:
            loc_list[0] = False  # 没找到设定标记，交给yolo
        else:
            loc_list[0] = bottle_loc  # 找到创建一个列表

        face_loc = self.get_face_loc(img)
        if not face_loc:
            loc_list[1] = False
        else:
            loc_list[1] = face_loc

        # print(False in loc_list)

        if False in loc_list:  # 实际上就是减少后台数据处理的消耗时间
            loc_data = self.get_all_loc(img)
            if not loc_data:
                return loc_list  # yolo都检测不到直接返回
            else:
                classes_list = ['bottle', 'face']
                for i in range(len(loc_list)):
                    if not loc_list[i]:
                        loc_list[i] = []

                for data_list in loc_data:
                    index = classes_list.index(data_list[0])
                    # ( (left, top), (right, end) )
                    loc_list[index].append(( (data_list[2], data_list[3]), (data_list[4], data_list[5]) ))
        return loc_list

    def get_bottle_loc(self, img: np.ndarray):
        """

        通过Svm的方案得到瓶子的位置
        然后根据返回值来确定是否调用yolo

        :param img:
        :return:
        """
        bottle_loc = self.bottle_locate_svm.predict(img)  # 返回定位数据
        # bottle_loc = self.bottle_locate_svm.predict_show(img)  # 用来测试直接返回图像
        return bottle_loc

    def get_face_loc(self, img: np.ndarray):
        face_loc = self.cascade.detect_face(img)
        # img = self.cascade.plot_rect(img)
        return face_loc

    def get_all_loc(self,
                    img: np.ndarray):
        predict_list = self.yolo_locate.predict(img)
        # predict_list = np.array(self.yolo_locate.draw(img, filter=None, font_path="../../Resource/model_data/simhei.ttf"))
        if predict_list:
            return predict_list[4]  # predicted_class, label, top, left, bottom, right
        else:
            return None


"""______________测试代码___________________"""


def get_pic():
    video = cv.VideoCapture(1)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    _object = ObjectLocate()

    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for pic in get_pic():
        pic = cv.resize(pic, (800, 600))
        loc_list = _object.get_loc(pic)
        bottle_loc, face_loc = loc_list
        print(loc_list)
        if bottle_loc:
            for start, end in bottle_loc:
                cv.rectangle(pic, start, end, color=color_list[0], thickness=2)
            flag = True
        if face_loc:
            for start, end in face_loc:
                cv.rectangle(pic, start, end, color=color_list[1], thickness=2)
            flag = True

        cv.imshow("pic", pic)
        cv.waitKey(1)

        # cv.imshow("img", img)
        # cv.waitKey(1)
