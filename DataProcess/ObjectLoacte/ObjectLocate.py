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
                 yolo_wight="../../Resource/model_data/test_model/yolo/Epoch100-Total_Loss7.1096-Val_Loss12.4228.pth",
                 yolo_anchors="../../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../../Resource/model_data/infusion_classes.txt",
                 ):
        self.bottle_locate_svm = LocateRoI(svm_path)
        self.yolo_locate = YoLoLocate(yolo_wight, yolo_anchors, yolo_predict_class)
        self.cascade = Cascade()

    def get_loc(self, img: np.ndarray, focus_on_model: bool = True):
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

        :param focus_on_model: 由于dlib的定位效果还是较差，所以yolov3具有一票否决权
        :param img: np.ndarray格式
        :return: loc_list
        """
        loc_list = [True, True]  # 表示找没找到

        bottle_loc = self.get_bottle_loc(img)
        # bottle_loc = None
        if not bottle_loc:
            loc_list[0] = False  # 没找到设定标记，交给yolo
        else:
            loc_list[0] = bottle_loc  # 找到创建一个列表

        face_loc = self.get_face_loc(img)
        # face_loc = None
        if not face_loc:
            loc_list[1] = False
        else:
            loc_list[1] = face_loc

        # print(False in loc_list)

        if False in loc_list:  # 实际上就是减少后台数据处理的消耗时间
            loc_data = self.get_all_loc(img)
            if not loc_data:
                return self.reliable_detect(loc_list)  # yolo都检测不到直接返回
            else:
                classes_list = ['bottle', 'face']
                for i in range(len(loc_list)):
                    if not loc_list[i]:
                        loc_list[i] = []

                has_bottle = False  # yolo的一票否决权

                for data_list in loc_data:
                    index = classes_list.index(data_list[0])
                    if index == 0:
                        has_bottle = True
                    # ( (left, top), (right, end) )
                    loc_list[index].append(((data_list[2], data_list[3]), (data_list[4], data_list[5])))

                if focus_on_model and not has_bottle:
                    loc_list[0] = False  # 否定dlib的定位结果
                    print("dlib was rejected by yolov3")

        # 做多个结果的合并
        loc_list = self.roi_filter(loc_list)
        return self.reliable_detect(loc_list)  # 转换为可信结果

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
            # 1. 没有就直接赋值
            # 2. 有的话计算交叉域，如果交叉域较大，那么选取边界较大的一个作为边界
            if not max_width:
                max_width = (left, right)
            else:
                last_left, last_right = max_width
                # 1. 找出交叉域
                max_left = max([last_left, left])
                min_right = min([last_right, right])

                # 2. 判断两者是否无关
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
                # 1. 找出交叉域
                max_top = max(last_top, top)
                min_end = min(last_end, end)

                # 2. 判断是否无关
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
                if roi > 0.6:  # 说明重叠度很高，取范围大的
                    last_left, last_right = max_width
                    max_width = (min(last_left, left), max(last_right, right))
                    last_top, last_end = max_height
                    max_height = (min(last_top, top), max(last_end, end))
            except:
                pass

        # 全部循环完毕后返回结果
        if not max_width or not max_height:
            return False
        else:
            left, right = max_width
            top, end = max_height
            # ( (left, top), (right, end) )
            new_face_list.append(((left, top), (right, end)))
            return new_face_list

    def reliable_detect(self, loc_list: list):
        """实际使用过程中出现了负数， 所以需要过滤"""
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
        print("YoLo predict", predict_list)
        # predict_list = np.array(self.yolo_locate.draw(img, filter=None, font_path="../../Resource/model_data/simhei.ttf"))
        if predict_list:
            return predict_list[4]  # predicted_class, label, top, left, bottom, right
        else:
            return None


"""______________测试代码___________________"""


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
