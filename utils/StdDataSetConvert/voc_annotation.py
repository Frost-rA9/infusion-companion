"""voc_annotation

    - 文件夹格式
    - Annotations:
        - 存放标签文件
        - 关注内容里的<object>标签
            1. 物体种类
            2. 物体位置
    - JPEGImages:
        - 存放标签文件对应的图片
    - ImageSets:
        - train.txt: 标签文件和图片文件除去后缀的名称
        -

    - 本类用来生成voc年份对应的train.txt
        1. 图片文件的路径
        2. 用逗号分割出5个数字
            - 前四个是目标位置
            - 最后一个是种类
        3. 多个目标信息用空格分割

    - 训练自己的网络步骤：
        1. classes改成自己的类
        2. 
"""

# ---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
# ---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# -----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
# -----------------------------------------------------#
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["bottle", "face"]

def convert_annotation(year, image_id, list_file):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    in_file = open('F:/DataSet/bottle/Locate/Annotations/%s.xml' % (image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()
"""官方代码"""
# for year, image_set in sets:
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set),
#                      encoding='utf-8').read().strip().split()
#     list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()

"""自己改的"""
# ('2007', 'train'), ('2007', 'val'), ('2007', 'test')
image_ids = open("F:/DataSet/bottle/Locate/ImageSets/train.txt", encoding="utf-8").read().strip().split()
list_file = open("model_train.txt", 'w', encoding="utf-8")
for image_id in image_ids:
    list_file.write("F:/DataSet/bottle/Locate/pic/%s.jpg" % (image_id))
    convert_annotation(0, image_id, list_file)
    list_file.write("\n")
list_file.close()