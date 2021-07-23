# encoding=GBK
"""SvmLabelToYoLo3

    - 由于svm标记的数据和yolo的定位数据大致一致，所以可以转换
"""

svm_path = "H:/bottle_black.xml"
xml_file = open(svm_path, encoding="utf-8")
yolo_train_txt = open("./YoLo_train.txt", mode="w")
id = {
    'bottle': 0,
    'face': 1
}

from bs4 import BeautifulSoup

bs = BeautifulSoup(xml_file, "lxml")
for images in bs.find_all("images"):
    for image in images.find_all("image"):
        # 1. 文件名
        file_path = image.attrs["file"]
        write_line = file_path
        for box in image.find_all('box'):
            # 2. 其中的一个坐标
            box_attrs = box.attrs
            left, top = box_attrs['left'], box_attrs['top']
            right, bottom = str(int(left) + int(box_attrs['width'])), str(int(top) + int(box_attrs['height']))
            # 3. 获得标签
            label = box.find('label').text
            write_line += " " + ",".join([left, top, right, bottom]) + "," + str(id[label])
        write_line += "\n"
        yolo_train_txt.write(write_line)

# if __name__ == '__main__':
# img = "F:/DataSet/bottle/Locate/JPEGImages/99.png"
# import cv2 as cv
# l = [333, 194, 478, 460]
# img = cv.imread(img)
# left, top, right, bottom = l
# img = cv.rectangle(img, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
# cv.imshow("img", img)
# cv.waitKey(0)
