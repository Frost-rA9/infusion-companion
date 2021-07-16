# encoding=GBK
"""DynamicThresholdDetection

    - 动态阈值检测
        - 负责用传统视觉的方式对目标进行检测
        - 由于二值化的阈值不好取，我们取5, 10, 15, 20的阈值分别进行液位检测
            - 最后把液位的值进行求平均得到一个相对合理的值

        - 本方案主要根据拍摄好数据集的经验值
            - 如果可以改成自适应的还好说
            - 否则还是按照一定比重跟模型进行混合

        - 现在此方案的误差：
            - 8-15%

    - cal_seq:
        - 按照以下次序执行：
        1. 去除图形文字干扰
        2. 进一步提取瓶子位置
        3. 对roi的瓶子再次进行分割，把有水的值设定为2，没有的设定为1
        4. 生成和img同shape但是没通道的图像，把roi的以外区域标记为0
        5. 计算液位水平

    - get_liquid_roi:
        - 和get_bottle_roi类似的操作方式
        - 主要变化在把bottle没水的地方设定为1
        - 有水的地方设定为2

    - drop_noisy:
        - 使用一下步骤去除图像中的文字（颜色）干扰项
        1. 到hsv空间
        2. 取饱和度通道
        3. 通过一个预设定的阈值进行全局二值化
        4. 通过膨胀操作，把文字部分的区域扩大
        5. 通过cv.inpaint对图像文字部分用周围像素进行替换

    - get_bottle_roi:
        - 使用以下步骤定位bottle_roi:
        1. 进行滤波
        2. 提取边缘
        3. 添加黑条屏蔽图片边界
        4. 腐蚀细节
        5. 提取最大边框

    - IQR:
        - 主要使用IQR对离群点进行过滤
        - 在多次滤波后通常指挥留下核心区域，但是上方通常还有一些离散点，
            - 通常是线缆的干扰
            - 所以把测试下来大约在1.2-1.3的过滤区间较好
            - 所以设定为1.2
        - 保留点的内容：返回 > Q1 - 1.5 * IQR的第一个值 还有 < Q3 + 1.5 * IQR 的第一个值
"""
import cv2 as cv
import numpy as np
from utils.ImageConvert.ConvertColorSpace import ConvertColorSpace
from utils.Caculate.LiquidLeftCal import LiquidLeftCal


class DynamicThresholdDetection:
    def __init__(self):
        self.binary_threshold = [5, 10, 15, 20]
        self.liquid_cal = LiquidLeftCal()

    def cal_seq(self, img: np.ndarray):
        # 1. 去除图形文字干扰
        img = self.drop_noise(img, threshold=self.binary_threshold[3])

        # 2. 进一步提取瓶子位置
        bottle_loc = self.get_bottle_roi(img)
        if not bottle_loc:
            print("DynamicThresholdDetection: undetected")
            return None

        left, top, right, bottom = bottle_loc
        bottle_roi = img[top: bottom, left: right]

        # 3. 对roi的瓶子再次进行分割，把有水的值设定为2，没有的设定为1
        liquid_roi = self.get_liquid_roi(bottle_roi)
        if not liquid_roi:
            print("DynamicThresholdDetection: undetected")
            return None

        left, top, right, bottom = liquid_roi
        bottle_roi[top: bottom, left: right] = 2  # 有水的地方值设定为2
        start_loc = 60
        bottle_roi[start_loc: top, left: right] = 1  # 没水的地方设定为1, 80是经验值，必须调整
        temp_roi = np.zeros(bottle_roi.shape, dtype=np.uint8)
        temp_roi[start_loc: bottom, left: right] = bottle_roi[start_loc: bottom, left: right]

        # 4. 生成和img同shape但是没通道的图像，把roi的以外区域标记为0
        background = np.zeros(img.shape, np.uint8)
        left, top, right, bottom = bottle_loc
        background[top: bottom, left: right] = temp_roi

        # 5. 返回液位水平
        data = self.liquid_cal.get_cur_liquid(background[:, :, 0])  # 每个通道值都一样
        return data

    def get_liquid_roi(self, img: np.ndarray):
        # 1. 加强对比度
        img = cv.normalize(img, dst=None, alpha=350, beta=10, norm_type=cv.NORM_MINMAX)

        # 2. 用sobel算子进行垂直梯度计算
        grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
        gradx = cv.convertScaleAbs(grad_x)
        grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

        # 3. 滤波减少噪点
        gaussian = cv.GaussianBlur(gradxy, ksize=(5, 5), sigmaX=0)

        # 4. 到灰度图，进行二值化
        gray = ConvertColorSpace.bgr_and_gray(to_gray=True, img=gradxy)
        ret, threshold_img = cv.threshold(gray, 48, 255, cv.THRESH_BINARY)  # 这里的阈值也要调整

        # 5. 先膨胀保护横向信息，在进行腐蚀
        element = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
        dilate = cv.morphologyEx(threshold_img, cv.MORPH_DILATE, element)

        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
        erode = cv.morphologyEx(dilate, cv.MORPH_ERODE, element)

        # 6. 边缘检测
        canny_img = cv.Canny(erode, 30, 15)

        # 7. 进行统计
        canny_img[canny_img == 255] = 1
        height_list, width_list = [], []
        for i in range(canny_img.shape[0]):
            temp = sum(canny_img[i])
            if temp != 0:
                height_list.append(i)
        for i in range(canny_img.shape[1]):
            temp = sum(canny_img[:, i])
            if temp != 0:
                width_list.append(i)

        height_loc = self.IQR(height_list, between=(-0.2, 0.2))
        width_loc = self.IQR(width_list)

        if height_loc and width_loc:
            top, bottom = height_loc
            left, right = width_loc
            return left, top, right, bottom
        else:
            return None

    def get_bottle_roi(self, img: np.ndarray):
        # 1. 进行滤波
        blur = cv.blur(img, ksize=(5, 5))

        # 2. 提取边缘
        canny_img = cv.Canny(blur, 30, 15)

        # 3. 添加黑条屏蔽图片边界
        canny_img[:, :10] = 0  # 上边框
        canny_img[:, -10:] = 0  # 下边框
        canny_img[:10, :] = 0  # 做边框
        canny_img[-10:, :] = 0  # 右边框

        # 4. 腐蚀细节
        element = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
        erode = cv.morphologyEx(canny_img, cv.MORPH_ERODE, element)

        element = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
        dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, element)  # 膨胀一次减少腐蚀损失

        element = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
        erode = cv.morphologyEx(dilate, cv.MORPH_ERODE, element)

        # 5. 提取最大边框
        height, width = erode.shape
        cal_map = (erode == 255).astype(np.int)
        width_list = []
        height_list = []

        for i in range(height):
            data = sum(cal_map[i])
            if data != 0:
                height_list.append(i)

        for j in range(width):
            data = sum(cal_map[:, j])
            if data != 0:
                width_list.append(j)

        height_loc = self.IQR(height_list)
        width_loc = self.IQR(width_list)

        if height_loc and width_loc:
            top, bottom = height_loc
            left, right = width_loc
            return left, top, right, bottom
        else:
            return None

    def IQR(self, data_list: list, between: tuple = (1.2, 1.2)):
        """IQR
            - 使用IQR的方法删除离群点
            1. 计算三个分位点的值[25, 50, 75]
            2. IQR = Q3 ? Q1
            3. 返回 > Q1 - 1.5 * IQR的第一个值 还有 < Q3 + 1.5 * IQR 的第一个值
        :param between: 离群点边界
        :param data_list: 需要过滤的列表
        :return: (start_value, end_value)
        """
        if data_list:
            list_len = len(data_list)
            index = [int(list_len * 0.25), int(list_len * 0.5), int(list_len * 0.75)]
            IQR = data_list[index[2]] - data_list[index[0]]
            lower = data_list[index[0]] - between[0] * IQR
            upper = data_list[index[2]] + between[1] * IQR
            start, end = 0, list_len - 1
            for i in range(list_len):
                if data_list[i] > lower:
                    start = i
                    # print("start is", start)
                    break
                else:
                    pass
            for i in range(list_len - 1, -1, -1):
                if data_list[i] < upper:
                    end = i
                    # print("end is", end)
                    break
                else:
                    pass

            if start > end:
                return None
            return data_list[start], data_list[end]
        else:
            return None

    def drop_noise(self, img: np.ndarray, threshold: int):
        # 1. 到hsv空间
        hsv = ConvertColorSpace.bgr_and_hsv(to_hsv=True, img=img)

        # 2. 提取饱和度图层
        # 色相/饱和度/明度
        s = hsv[:, :, 1]

        # 3. 全局二值化
        th = threshold
        ret, threshold_img = cv.threshold(s, th, 1, cv.THRESH_BINARY)

        # 4. 形态学操作
        # 膨胀把小框框膨胀成大框框
        element = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
        dilate = cv.morphologyEx(threshold_img, cv.MORPH_DILATE, element)  # 膨胀操作

        # 5. 把膨胀后的结果作为掩膜
        # 掩膜会把文字部分替替换成周围的颜色，从而消除干扰项
        inpaint_img = cv.inpaint(img, dilate, 3, cv.INPAINT_TELEA)

        return inpaint_img


"""这里是测试代码"""


def show_img(img: np.ndarray):
    cv.imshow("img", img)
    cv.waitKey(0)


if __name__ == '__main__':
    from PIL import Image
    import os
    file_path = "F:/DataSet/bottle/segmentation/dir_json/train/"
    dynamic = DynamicThresholdDetection()
    liquid = LiquidLeftCal()
    np.set_printoptions(threshold=np.inf)

    total = []
    for d in os.listdir(file_path):
        temp = file_path + d + "/"
        img_path = temp + "img.png"
        img = cv.imread(img_path)
        data1 = dynamic.cal_seq(img)  # 0.6164383561643836

        img_path = temp + "label.png"
        i = Image.open(img_path)
        i = np.array(i)
        data2 = liquid.get_cur_liquid(i)
        print("*" * 20)
        print("data1", data1, "data2", data2)
        differ = abs(data1 - data2) / data2 * 100
        if differ != np.inf:
            total.append(differ)
        print("differ rate is:", differ)
        print("*" * 20)

    print(sum(total) / len(total))
