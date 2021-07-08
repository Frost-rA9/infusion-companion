# encoding=GBK
"""Decode

    - DecodeBox

        - channel�ĺ���: 3(�����) * [1(�����Ƿ�������) * 4(�����ĵ�������) + num_classes(��������)]
        - ê��
          # -----------------------------------------------------------#
          #   13x13���������Ӧ��anchor��[116,90],[156,198],[373,326]
          #   26x26���������Ӧ��anchor��[30,61],[62,45],[59,119]
          #   52x52���������Ӧ��anchor��[10,13],[16,30],[33,23]
          # -----------------------------------------------------------#

"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # -----------------------------------------------#
        #   �����inputһ�������������ǵ�shape�ֱ���
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        # -----------------------------------------------#
        batch_size = input.size(0)  # ������ͼ��
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   ����Ϊ416x416ʱ
        #   stride_h = stride_w = 32��16��8
        #   ʵ���Ͼ��ǻ��ָ��ӣ���������Ϊh * w
        # -----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        # -------------------------------------------------#
        #   ��ʱ��õ�scaled_anchors��С��������������
        #   ���������еȱ�������
        #   ����416*416��ê��/32�Ϳ������ŵ��������(13,13)��С��
        # -------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]

        # -----------------------------------------------#
        #   �����inputһ�������������ǵ�shape�ֱ���
        #   batch_size, 3, 13, 13, 75
        #   batch_size, 3, 26, 26, 75
        #   batch_size, 3, 52, 52, 75
        #   view-> 3 * ( 5 + num_classes)
        #   self.num_anchors: 3, self.bbox_attrs: 5+num_classes
        #   permute������ά��,��5+num_classes���������һ��ά��
        # -----------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # ����������λ�õĵ�������
        # sigmoid���ǰ���ֵ���ŵ�[0-1]
        # (left, top, width, height, available)
        x = torch.sigmoid(prediction[..., 0])  # (5+num_classes)�ĵ�һά��
        y = torch.sigmoid(prediction[..., 1])
        # �����Ŀ�ߵ�������
        w = prediction[..., 2]  # width
        h = prediction[..., 3]  # height
        # ������Ŷȣ��Ƿ�������
        conf = torch.sigmoid(prediction[..., 4])
        # num_classes�ĸ���
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------#
        #   ��������ģ��������Ͻ�, ÿ�������ߵĽ���
        #   batch_size,3,13,13
        # ----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # ----------------------------------------------------------#
        #   ���������ʽ���������Ŀ��
        #   batch_size,3,13,13
        # ----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   ����Ԥ�������������е���
        #   ���ȵ������������ģ�����������������½�ƫ��
        #   �ٵ��������Ŀ�ߡ�
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # ----------------------------------------------------------#
        #   �����������������������ͼ���С(416, 416)
        # ----------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data  # �����������λ�������


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        ����IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # ----------------------------------------------------------#
    #   ��Ԥ�����ĸ�ʽת�������Ͻ����½ǵĸ�ʽ��
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   ������Ԥ�ⲿ��ȡmax��
        #   class_conf  [num_anchors, 1]    �������Ŷ�
        #   class_pred  [num_anchors, 1]    ����
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   �������ŶȽ��е�һ��ɸѡ
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        # ----------------------------------------------------------#
        #   �������ŶȽ���Ԥ������ɸѡ
        # ----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7������Ϊ��x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # ------------------------------------------#
        #   ���Ԥ�����а�������������
        # ------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # ------------------------------------------#
            #   ���ĳһ��÷�ɸѡ��ȫ����Ԥ����
            # ------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            # ------------------------------------------#
            #   ʹ�ùٷ��Դ��ķǼ������ƻ��ٶȸ���һЩ��
            # ------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]

            # # ���մ�����������Ŷ�����
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # ���зǼ�������
            # max_detections = []
            # while detections_class.size(0):
            #     # ȡ����һ�����Ŷ���ߵģ�һ��һ�������жϣ��ж��غϳ̶��Ƿ����nms_thres���������ȥ����
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # �ѵ�
            # max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
