# encoding=GBK
"""yolo

    - YOLO:
        1. ��ȡ���еķ���
        2. ����yolo��Ȩ��
        3. detect_image:
            - ��ͼ�����Ԥ�⣬���������

    - YoLoLocate:
        1. ��ʼ��YOLO
        2. �ָ�YOLO.detect_image:
            - predict: ����Ԥ���
            - draw: ���ػ��ƺõ�ͼ�񣬻�������Ϊfilter�������
"""

# -------------------------------------#
#       ����YOLO��
# -------------------------------------#
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from net.YOLOV3.YOLOV3 import YOLOV3 as YoloBody
from net.YOLOV3.utils.Decode import DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes


# --------------------------------------------#
#   ʹ���Լ�ѵ���õ�ģ��Ԥ����Ҫ�޸�2������
#   model_path��classes_path����Ҫ�޸ģ�
#   �������shape��ƥ�䣬һ��Ҫע��
#   ѵ��ʱ��model_path��classes_path�������޸�
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": '../../Resource/model_data/yolo_weights.pth',
        "anchors_path": '../../Resource/model_data/yolo_anchors.txt',
        "classes_path": '../../Resource/model_data/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True,
        # ---------------------------------------------------------------------#
        #   �ñ������ڿ����Ƿ�ʹ��letterbox_image������ͼ����в�ʧ���resize��
        #   �ڶ�β��Ժ󣬷��ֹر�letterbox_imageֱ��resize��Ч������
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   ��ʼ��YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   ������еķ���
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   ������е������
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   ����ģ��
    # ---------------------------------------------------#
    def generate(self):
        self.num_classes = len(self.class_names)
        # ---------------------------------------------------#
        #   ����yolov3ģ��
        # ---------------------------------------------------#
        self.net = YoloBody(self.anchors, self.num_classes)

        # ---------------------------------------------------#
        #   ����yolov3ģ�͵�Ȩ��
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # ---------------------------------------------------#
        #   ������������������õĹ���
        # ---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], self.num_classes, (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # �������ò�ͬ����ɫ
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   ���ͼƬ
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   �����ｫͼ��ת����RGBͼ�񣬷�ֹ�Ҷ�ͼ��Ԥ��ʱ����
        # ---------------------------------------------------------#
        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   letterbox�����ã�
        #       - ��ͼ�����ӻ�����ʵ�ֲ�ʧ���resize
        #       - Ҳ����ֱ��resize����ʶ��
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)

        photo = np.array(crop_img, dtype=np.float32) / 255.0  # ��һ��
        photo = np.transpose(photo, (2, 0, 1))  # ͨ�����任
        # ---------------------------------------------------------#
        #   �����batch_sizeά��
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))  # ��tensor
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #   ��ͼ���������統�н���Ԥ�⣡
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            # ��������Ԥ�������л���
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #   ��Ԥ�����жѵ���Ȼ����зǼ�������
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.num_classes, conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            # ---------------------------------------------------------#
            #   ���û�м������壬����ԭͼ
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # ---------------------------------------------------------#
            #   ��Ԥ�����е÷�ɸѡ
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   ��ͼ��������Ԥ��ǰ�����letterbox_image��ͼ����Χ��ӻ���
            #   ������ɵ�top_bboxes��������л�����ͼ���
            #   ������Ҫ��������޸ģ�ȥ�������Ĳ��֡�
            # -----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

            return image, self.model_image_size, top_label, top_conf, boxes, self.colors, self.class_names

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   ��ͼ�����ӻ�����ʵ�ֲ�ʧ���resize
        #   Ҳ����ֱ��resize����ʶ��
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   �����batch_sizeά��
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detections[top_index, :4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
            except:
                pass

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names),
                                                       conf_thres=self.confidence,
                                                       nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                   image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
                except:
                    pass

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    @property
    def defaults(self):
        return self._defaults


class YoLoLocate:
    print_var = False  # ���������Ƿ��ӡ�м���Ϣ

    def __init__(self, model_path: str, anchors_path: str, classes_path: str):
        # ����python�����·���Ǵӵ�ǰ�����ļ������ģ�������Ҫ����
        temp = {"model_path": model_path, "anchors_path": anchors_path, "classes_path": classes_path}
        for key in temp:
            YOLO._defaults[key] = temp[key]
        self.yolo_detect = YOLO()

    def predict(self, img: np.ndarray):
        img = Image.fromarray(img)  # to PIL
        try:
            image, model_image_size, top_label, top_conf, boxes, colors, class_names = self.yolo_detect.detect_image(
                img)
        except:
            return None
        predict_list = []  # Ԥ�����ݴ�

        for i, c in enumerate(top_label):
            # print(c)
            predicted_class = class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)

            predict_list.append((predicted_class, label, left, top, right, bottom))
        return image, model_image_size, colors, class_names, predict_list

    def draw(self, img: np.ndarray, filter: list, font_path: str):
        try:
            image, model_image_size, colors, class_names, predict_list = self.predict(img)
        except:
            return Image.fromarray(img)
        font = ImageFont.truetype(font=font_path,
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // model_image_size[0], 1)

        for predicted_class, label, left, top, right, bottom in predict_list:
            if filter:  # filter����ʱ�Ž��й���
                if not label.split()[0] in filter:
                    pass

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if YoLoLocate.print_var:
                print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


"""-------���Դ���--------"""


def get_pic():
    video = cv.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    from PIL import Image
    import cv2 as cv
    locate = YoLoLocate("../../Resource/model_data/test_model/yolo/Epoch100-Total_Loss6.6752-Val_Loss11.2832.pth",
                        "../../Resource/model_data/yolo_anchors.txt",
                        "../../Resource/model_data/infusion_classes.txt")

    for frame in get_pic():
        predict = locate.draw(frame, None, "../../Resource/model_data/simhei.ttf")
        if predict:
            frame = np.array(predict)
        cv.imshow("frame", frame)
        cv.waitKey(1)

    # r_img = yolo.detect_image(img)
    # r_img.show()
