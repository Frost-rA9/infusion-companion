# encoding=GBK
"""DynamicThresholdDetection

    - ��̬��ֵ���
        - �����ô�ͳ�Ӿ��ķ�ʽ��Ŀ����м��
        - ���ڶ�ֵ������ֵ����ȡ������ȡ5, 10, 15, 20����ֵ�ֱ����Һλ���
            - ����Һλ��ֵ������ƽ���õ�һ����Ժ����ֵ

        - ��������Ҫ������������ݼ��ľ���ֵ
            - ������Ըĳ�����Ӧ�Ļ���˵
            - �����ǰ���һ�����ظ�ģ�ͽ��л��

        - ���ڴ˷�������
            - 8-15%

    - cal_seq:
        - �������´���ִ�У�
        1. ȥ��ͼ�����ָ���
        2. ��һ����ȡƿ��λ��
        3. ��roi��ƿ���ٴν��зָ����ˮ��ֵ�趨Ϊ2��û�е��趨Ϊ1
        4. ���ɺ�imgͬshape����ûͨ����ͼ�񣬰�roi������������Ϊ0
        5. ����Һλˮƽ

    - get_liquid_roi:
        - ��get_bottle_roi���ƵĲ�����ʽ
        - ��Ҫ�仯�ڰ�bottleûˮ�ĵط��趨Ϊ1
        - ��ˮ�ĵط��趨Ϊ2

    - drop_noisy:
        - ʹ��һ�²���ȥ��ͼ���е����֣���ɫ��������
        1. ��hsv�ռ�
        2. ȡ���Ͷ�ͨ��
        3. ͨ��һ��Ԥ�趨����ֵ����ȫ�ֶ�ֵ��
        4. ͨ�����Ͳ����������ֲ��ֵ���������
        5. ͨ��cv.inpaint��ͼ�����ֲ�������Χ���ؽ����滻

    - get_bottle_roi:
        - ʹ�����²��趨λbottle_roi:
        1. �����˲�
        2. ��ȡ��Ե
        3. ��Ӻ�������ͼƬ�߽�
        4. ��ʴϸ��
        5. ��ȡ���߿�

    - IQR:
        - ��Ҫʹ��IQR����Ⱥ����й���
        - �ڶ���˲���ͨ��ָ�����º������򣬵����Ϸ�ͨ������һЩ��ɢ�㣬
            - ͨ�������µĸ���
            - ���԰Ѳ���������Լ��1.2-1.3�Ĺ�������Ϻ�
            - �����趨Ϊ1.2
        - ����������ݣ����� > Q1 - 1.5 * IQR�ĵ�һ��ֵ ���� < Q3 + 1.5 * IQR �ĵ�һ��ֵ
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
        # 1. ȥ��ͼ�����ָ���
        img = self.drop_noise(img, threshold=self.binary_threshold[3])

        # 2. ��һ����ȡƿ��λ��
        bottle_loc = self.get_bottle_roi(img)
        if not bottle_loc:
            print("DynamicThresholdDetection: undetected")
            return None

        left, top, right, bottom = bottle_loc
        bottle_roi = img[top: bottom, left: right]

        # 3. ��roi��ƿ���ٴν��зָ����ˮ��ֵ�趨Ϊ2��û�е��趨Ϊ1
        liquid_roi = self.get_liquid_roi(bottle_roi)
        if not liquid_roi:
            print("DynamicThresholdDetection: undetected")
            return None

        left, top, right, bottom = liquid_roi
        bottle_roi[top: bottom, left: right] = 2  # ��ˮ�ĵط�ֵ�趨Ϊ2
        start_loc = 60
        bottle_roi[start_loc: top, left: right] = 1  # ûˮ�ĵط��趨Ϊ1, 80�Ǿ���ֵ���������
        temp_roi = np.zeros(bottle_roi.shape, dtype=np.uint8)
        temp_roi[start_loc: bottom, left: right] = bottle_roi[start_loc: bottom, left: right]

        # 4. ���ɺ�imgͬshape����ûͨ����ͼ�񣬰�roi������������Ϊ0
        background = np.zeros(img.shape, np.uint8)
        left, top, right, bottom = bottle_loc
        background[top: bottom, left: right] = temp_roi

        # 5. ����Һλˮƽ
        data = self.liquid_cal.get_cur_liquid(background[:, :, 0])  # ÿ��ͨ��ֵ��һ��
        return data

    def get_liquid_roi(self, img: np.ndarray):
        # 1. ��ǿ�Աȶ�
        img = cv.normalize(img, dst=None, alpha=350, beta=10, norm_type=cv.NORM_MINMAX)

        # 2. ��sobel���ӽ��д�ֱ�ݶȼ���
        grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
        gradx = cv.convertScaleAbs(grad_x)
        grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

        # 3. �˲��������
        gaussian = cv.GaussianBlur(gradxy, ksize=(5, 5), sigmaX=0)

        # 4. ���Ҷ�ͼ�����ж�ֵ��
        gray = ConvertColorSpace.bgr_and_gray(to_gray=True, img=gradxy)
        ret, threshold_img = cv.threshold(gray, 48, 255, cv.THRESH_BINARY)  # �������ֵҲҪ����

        # 5. �����ͱ���������Ϣ���ڽ��и�ʴ
        element = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
        dilate = cv.morphologyEx(threshold_img, cv.MORPH_DILATE, element)

        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
        erode = cv.morphologyEx(dilate, cv.MORPH_ERODE, element)

        # 6. ��Ե���
        canny_img = cv.Canny(erode, 30, 15)

        # 7. ����ͳ��
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
        # 1. �����˲�
        blur = cv.blur(img, ksize=(5, 5))

        # 2. ��ȡ��Ե
        canny_img = cv.Canny(blur, 30, 15)

        # 3. ��Ӻ�������ͼƬ�߽�
        canny_img[:, :10] = 0  # �ϱ߿�
        canny_img[:, -10:] = 0  # �±߿�
        canny_img[:10, :] = 0  # ���߿�
        canny_img[-10:, :] = 0  # �ұ߿�

        # 4. ��ʴϸ��
        element = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
        erode = cv.morphologyEx(canny_img, cv.MORPH_ERODE, element)

        element = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
        dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, element)  # ����һ�μ��ٸ�ʴ��ʧ

        element = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
        erode = cv.morphologyEx(dilate, cv.MORPH_ERODE, element)

        # 5. ��ȡ���߿�
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
            - ʹ��IQR�ķ���ɾ����Ⱥ��
            1. ����������λ���ֵ[25, 50, 75]
            2. IQR = Q3 ? Q1
            3. ���� > Q1 - 1.5 * IQR�ĵ�һ��ֵ ���� < Q3 + 1.5 * IQR �ĵ�һ��ֵ
        :param between: ��Ⱥ��߽�
        :param data_list: ��Ҫ���˵��б�
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
        # 1. ��hsv�ռ�
        hsv = ConvertColorSpace.bgr_and_hsv(to_hsv=True, img=img)

        # 2. ��ȡ���Ͷ�ͼ��
        # ɫ��/���Ͷ�/����
        s = hsv[:, :, 1]

        # 3. ȫ�ֶ�ֵ��
        th = threshold
        ret, threshold_img = cv.threshold(s, th, 1, cv.THRESH_BINARY)

        # 4. ��̬ѧ����
        # ���Ͱ�С������ͳɴ���
        element = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
        dilate = cv.morphologyEx(threshold_img, cv.MORPH_DILATE, element)  # ���Ͳ���

        # 5. �����ͺ�Ľ����Ϊ��Ĥ
        # ��Ĥ������ֲ������滻����Χ����ɫ���Ӷ�����������
        inpaint_img = cv.inpaint(img, dilate, 3, cv.INPAINT_TELEA)

        return inpaint_img


"""ThresholdDetect

    - ��opencv���Һλ�ĵ����汾
    - ��Ҫ��̬�ĵ���������ֵ��������Ҫ��ķ���
    
    - ���̸������£�
        1. ���涶���жϣ�
            - ���������ģ�����ݼ��������ж�ģ��ͼ��Ķ�λ����
            ����ģ��ͼ�ζԴ�ͳ�����ĸ��Ź���
            �����ģ�����ݵ������ֱ�ӽ���ģ�ͼ���
            - �����ʵ�ֺ������Ϊ��̬������Process.py���׼ȷ��
            - ��ʵ�ּ򵥵ķ�����JitterCal.py��
                - ���ڶ�����Ҫǰ������ͼ���бȽϣ�������Ҫ�и����ѭ����������
        2. ͨ����ʴ���Ͳ���ȥ��������̶�
            - ����ͨ��HSV��ɫ��ռ䣬��λ�����ֺ�ƿ��
            - Ȼ��ͨ������˵����Ͳ����������ֵ�����
            - Ȼ��ʹ��cv.inpaint����Χ���ؽ������
            - �����Ӿ��ܴ�Χ��ȥ��ͼ��ĸ�������
        3. ֱ��ͼ����
            - ����ͳ�ƻҶ�ͼ��ÿ������ֵ�ĸ���
            - Ȼ���ÿһ��/(width * height) ����ֵȫ��ƽ����
            - Ȼ����������ص��ֵ����Ӧ�ĸ��ʽ����滻
        4. ��ֵ�ָ�
            - ֱ��ͼ�����Ľ�����ڱ߽��ֵ��Խ�С
            - ����ʹ�ö�ֵ����ͨ���ᱻ��ֵ��Ϊ0
            - �����Ӽ����������еļ����
                - Ȼ�����ÿ�еĲ�ֵ���ܵõ��ݶ�
        5. �߽���
            - �ҵ���ֵ���ĵط���һ����Ǳ߽���
            
        
"""
from utils.Caculate.JitterCal import JitterCal


class ThresholdDetect2:
    print_var = False

    def __init__(self):
        self.liquid_cal = LiquidLeftCal()  # ���ڷָ�Ľ��������

    def cal_seq(self, img: np.ndarray):
        if ThresholdDetect2.print_var:
            show_img(img)

        # 1. ��ͼ������ֽ���ģ��
        img = self.drop_noise(img)

        # 2. ��ģ��ͼ�����Ԥ��
        img, start_pos, end_pos = self.predict(img)

        # 3. ֱ��ͨ�����¸߶ȱ����õ�Һλֵ
        img = cv.line(img, start_pos, end_pos, color=(0, 255, 0), thickness=2)

        if ThresholdDetect2.print_var:
            show_img(img)

        left, top = start_pos
        height, width = img.shape
        return (height - top) / height

    def predict(self, img: np.ndarray):
        # 1. ����Ҷ�ֱ��ͼ
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # ���Ҷ�ͼ
        gray_count = np.zeros((256), np.int)
        flatten_img = img.flatten()
        for i in range(len(flatten_img)):
            gray_count[flatten_img[i]] += 1  # ͳ����ÿ������ֵ��ĸ���

        # 2. ֱ��ͼ���⻯
        height, width = img.shape
        gray_count_per = gray_count / (width * height)  # ÿ������ƽ��
        for i in range(2, len(gray_count_per)):  # ��֪��Ϊʲô��2��ʼ
            gray_count_per[i] = gray_count_per[i - 1] + gray_count_per[i]

        # 3. ����Ҷȸ���
        gray_count_per = gray_count_per * 255
        buff = np.zeros((height * width), np.double)
        for i in range(len(flatten_img)):  # ÿ������ֵ�ø����滻
            buff[i] = gray_count_per[flatten_img[i]]
        buff = np.reshape(buff, (height, width))  # �ѳ��������ͼ��

        # 4. ����3x3�ľ�ֵ�˲�
        blur = cv.blur(buff, ksize=(3, 3))

        # 5. ͼ�ζ�ֵ��
        """ͬ����Ҫ����ʵ���������"""
        threshold = 120
        ret, threshold_img = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)

        if ThresholdDetect2.print_var:
            show_img(threshold_img)

        # 6. ����ÿ��������ֵ��ӵĺ�
        start_loc = 10  # ���˵���Һƿ��ͷ��
        sum_list = []
        for i in range(start_loc, height, 2):  # ����ż�еĺ�, ������ҪΪż��
            try:
                sum_list.append(sum(threshold_img[i]) + sum(threshold_img[i + 1]))
            except:
                pass  # ֱ�Ӻ���һ��
        # 7. ����sum_list��ÿ���еĲ�
        sub_list = []
        for i in range(0, len(sum_list) - 1):
            try:
                sub_list.append(abs(sum_list[i + 1] - sum_list[i]))
            except:
                pass  # ͬ����ֹ�����ı���

        # 8. �ҳ��Աȶ�����λ��
        max_index = sub_list.index(max(sub_list))
        true_start_height = max_index * 2 + start_loc  # ���ǲ���Ϊ2����λ��

        # 9. ����
        # ���¾��ǻ���һ�������Աȶȿ�ʼ��һ����20���صĿ�
        start_pos = (0, true_start_height)
        end_pos = (width, true_start_height)
        # img = cv.rectangle(img, start_pos, end_pos, color=(0, 255, 0), thickness=2)
        # img = cv.line(img, start_pos, end_pos, color=(0, 255, 0), thickness=2)

        if ThresholdDetect2.print_var:
            show_img(img)

        return img, start_pos, end_pos

    def drop_noise(self, img: np.ndarray):
        # 1. ��hsv�ռ�
        hsv = ConvertColorSpace.bgr_and_hsv(to_hsv=True, img=img)

        # 2. ��ȡ���Ͷ�ͼ��
        # ɫ��/���Ͷ�/����
        s = hsv[:, :, 0]

        if ThresholdDetect2.print_var:
            show_img(s)

        # 3. ȫ�ֶ�ֵ��
        """���100��Ҫ���ݹ��յ���"""
        th = 100
        ret, threshold_img = cv.threshold(s, th, 255, cv.THRESH_BINARY)

        if ThresholdDetect2.print_var:
            show_img(threshold_img)

        # 4. ��̬ѧ����
        # ���Ͱ�С������ͳɴ���
        element = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
        dilate = cv.morphologyEx(threshold_img, cv.MORPH_DILATE, element)  # ���Ͳ���

        if ThresholdDetect2.print_var:
            show_img(dilate)

        # 5. �����ͺ�Ľ����Ϊ��Ĥ
        # ��Ĥ������ֲ������滻����Χ����ɫ���Ӷ�����������
        inpaint_img = cv.inpaint(img, dilate, 3, cv.INPAINT_TELEA)

        if ThresholdDetect2.print_var:
            show_img(inpaint_img)

        return inpaint_img


"""�����ǲ��Դ���"""


def show_img(img: np.ndarray, win_name: str = "img"):
    cv.imshow(win_name, img)
    cv.waitKey(0)


if __name__ == '__main__':
    from utils.ImageLoaderHelper.VideoHelper import VideoHelper
    ThresholdDetect2.print_var = True
    threshold2 = ThresholdDetect2()
    ### ThresholdDetect2 test code
    l = []
    less_than_30 = 0
    import os
    img_path = "F:/temp/Classifer/train/lower/"
    for file in os.listdir(img_path):
        img = img_path + file
        img = cv.imread(img)
        data = threshold2.cal_seq(img)
        if data < 0.4:
            less_than_30 += 1
        print(data)
        l.append(data)
    print("less than 40", less_than_30)
    print("ave data:", sum(l) / len(l))
    # last_frame = None
    # for frame in VideoHelper.read_frame_from_cap(0):
    #     if last_frame is not None:
    #         data = JitterCal.jitter_detect(last_frame, frame)
    #     last_frame = frame
    #     show_img(frame)

    ### for DynamicThresholdDetection test code
    # file_path = "F:/DataSet/bottle/segmentation/dir_json/train/"
    # dynamic = DynamicThresholdDetection()
    # liquid = LiquidLeftCal()
    # np.set_printoptions(threshold=np.inf)
    #
    # total = []
    # for d in os.listdir(file_path):
    #     temp = file_path + d + "/"
    #     img_path = temp + "img.png"
    #     img = cv.imread(img_path)
    #     data1 = dynamic.cal_seq(img)  # 0.6164383561643836
    #
    #     img_path = temp + "label.png"
    #     i = Image.open(img_path)
    #     i = np.array(i)
    #     data2 = liquid.get_cur_liquid(i)
    #     print("*" * 20)
    #     print("data1", data1, "data2", data2)
    #     differ = abs(data1 - data2) / data2 * 100
    #     if differ != np.inf:
    #         total.append(differ)
    #     print("differ rate is:", differ)
    #     print("*" * 20)
    #
    # print(sum(total) / len(total))
