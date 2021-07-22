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
    print_var = False  # 用来控制是否打印中间信息

    def __init__(self,
                 svm_path="../../Resource/svm/trained/new_bottle_svm.svm",
                 yolo_wight="../../Resource/model_data/test_model/yolo/Epoch100-Total_Loss7.1096-Val_Loss12.4228.pth",
                 yolo_anchors="../../Resource/model_data/yolo_anchors.txt",
                 yolo_predict_class="../../Resource/model_data/infusion_classes.txt",
                 ):
        self.bottle_locate_svm = LocateRoI(svm_path)
        self.yolo_locate = YoLoLocate(yolo_wight, yolo_anchors, yolo_predict_class)
        self.cascade = Cascade()
        # 格式( loc: (left, top, right, bottom), count: )
        # loc计算roi，且用于返回，应为瓶子大体是不移动的
        # count：每隔一定时间-1，变成0去除，如果一直大于3，那么说明这里确实是瓶子
        self.is_bottle = {}
        self.frame_down = False  # 刚好两种状态来判定什么时候减少一定的值

    """1种对外开放的主调用函数
        - 通过3种基本方法还有几种优化方案返回比较合理的定位
    """

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
                    if ObjectLocate.print_var:
                        print("dlib was rejected by yolov3")

        # 做多个结果的合并
        loc_list = self.roi_filter(loc_list)
        return self.reliable_detect(loc_list)  # 转换为可信结果

    """4种预测框过滤机制
        - 人脸定位：
            1. 由于opencv和yolo定位都比较准，但是框的大小不一致
                - 由于五官最好都包括进来，所以如果存在较大的roi就合并这复数个框
        - 瓶子定位：
            - 瓶子定位svm和yolo都挺飘，所以使用投票机制
            1. 每次定位更新self.is_bottle
                - 如果是空的直接加入
                - 否则逐个比较，计算roi，如果有超过70的，就直接把那个计数+1
                - 然后每2帧就把全部的数值减去1，如果有只为0的，就直接丢弃
                    1. 要么瓶子移走了，要么不是瓶子
                    2. 由于是每2帧减少1次，所以如果每次都有瓶子，总会超过3的
            2. 检测是否有超过3的位置，如果有就返回列表
            3. 检验框的干扰越小越好，所以框应该画小点
        - 人脸被预测为瓶子：
            - 暴力方案：
            1. 计算roi，如果超过20, 直接舍弃瓶子的框
    """

    def roi_filter(self, loc_list: list):
        bottle_list, face_list = loc_list

        bottle_list = self.get_possible_bottle_roi(bottle_list)
        face_list = self.get_max_face_roi(face_list)
        bottle_list = self.filter_bottle_to_face(bottle_list, face_list)
        bottle_list = self.get_max_face_roi(bottle_list, get_bigger=False, face_roi=0.3)
        loc_list[0] = bottle_list
        loc_list[1] = face_list
        return loc_list

    def filter_bottle_to_face(self, bottle_list: list, face_list: list, cross_area=0.05):
        if not bottle_list or not face_list:
            return bottle_list

        new_bottle_list = []
        for bottle_start, bottle_end in bottle_list:
            for face_start, face_end in face_list:
                rate = ObjectLocate.cal_cross_area((bottle_start, bottle_end), (face_start, face_end))
                if rate > cross_area:
                    break
            else:
                new_bottle_list.append((bottle_start, bottle_end))
        if len(new_bottle_list) == 0:
            return False
        else:
            return new_bottle_list

    def get_possible_bottle_roi(self,
                                bottle_list,
                                cross_rate: float = 0.8,
                                decrease_rate=0.15):
        """
        :param decrease_rate: 每2帧的下降水平
        :param bottle_list: 瓶子的定位数据
        :param cross_rate: 相似度
        :return:
        """
        if bottle_list is False:
            return False

        # 1. 计算roi增加计数
        for data in bottle_list:
            if not data:
                pass

            (left, top), (right, end) = data
            if len(self.is_bottle) == 0:
                self.is_bottle[((left, top), (right, end))] = 1
            else:
                roi_rate_list = []
                item_list = []  # 方便赋值
                for item in self.is_bottle:
                    roi_rate_list.append(ObjectLocate.cal_cross_area(((left, top), (right, end)), item))
                    item_list.append(item)
                    # 2. 看看是否需要相减, 写在这里减少消耗
                    if self.frame_down:
                        self.is_bottle[item] = self.is_bottle[item] - decrease_rate

                if ObjectLocate.print_var:
                    print("bottle roi_rate_list", roi_rate_list)

                max_index = roi_rate_list.index(max(roi_rate_list))
                if roi_rate_list[max_index] > cross_rate:
                    self.is_bottle[item_list[max_index]] = self.is_bottle.get(item_list[max_index], 0) + 1
                else:
                    self.is_bottle[((left, top), (right, end))] = 1

        # 刚好两种状态换来换去
        self.frame_down = not self.frame_down

        # 3. 看看是否要返回
        return_list = []  # 存放遍历过程中值大于3的item
        del_list = []
        for item in self.is_bottle:
            value = self.is_bottle[item]
            if value >= 3:
                return_list.append(item)

            if value < 0:  # 对流程做了一定更改，所以改成<0
                del_list.append(item)

        for del_item in del_list:
            self.is_bottle.pop(del_item)
        if ObjectLocate.print_var:
            print("self.is_bottle", self.is_bottle)

        if return_list:
            return return_list
        else:
            return False

    @staticmethod
    def cal_cross_area(loc1: tuple, loc2: tuple):
        (left_1, top_1), (right_1, end_1) = loc1
        (left_2, top_2), (right_2, end_2) = loc2

        # 1. 计算横向
        max_left = max([left_1, left_2])
        min_right = min([right_1, right_2])
        # 没有交集直接返回
        if min_right <= max_left:
            return 0
        horizontal_cross = min_right - max_left
        horizontal_area = max(right_1, right_2) - min(left_1, left_2)
        horizontal_roi = horizontal_cross / horizontal_area

        # 2. 计算纵向
        max_top = max(top_1, top_2)
        min_end = min(end_1, end_2)
        # 无关直接返回
        if max_top >= min_end:
            return 0
        vertical_cross = min_end - max_top
        vertical_area = max(end_1, end_2) - min(top_1, top_2)
        vertical_roi = vertical_cross / vertical_area

        return horizontal_roi * vertical_roi

    def get_max_face_roi(self, face_list: list, face_roi=0.6, get_bigger=True):
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
                    new_face_list.append(((left, top), (right, end)))
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
                if ObjectLocate.print_var:
                    print("Cross area roi is: ", roi)
                if roi > face_roi:  # 说明重叠度很高，取范围大的
                    if get_bigger:  # 代码复用，脸取大的，瓶子取小的
                        last_left, last_right = max_width
                        max_width = (min(last_left, left), max(last_right, right))
                        last_top, last_end = max_height
                        max_height = (min(last_top, top), max(last_end, end))
                    else:
                        last_left, last_right = max_width
                        max_width = (max(last_left, left), min(last_right, right))
                        last_top, last_end = max_height
                        max_height = (max(last_top, top), min(last_end, end))
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

    """1种输出格式规范化
        - 其实就是把输出结果整整合理
            1. 别出现负数
            2. 别出现 left > right, top > bottom的情况
    """

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

    """3种基础定位方法:
        1. svm
        2. Cascade
        3. yolo
    """

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
        if ObjectLocate.print_var:
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
