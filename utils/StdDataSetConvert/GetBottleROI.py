# encoding=GBK
import cv2 as cv
lower_txt = "H:/bottle_lower.txt"
bottle_txt = "H:/label/YoLo_train(only_bottle).txt"
img_dir = "H:/bottle"
lower_path = "H:/Classifer/train/lower/"
upper_path = "H:/Classifer/train/upper/"

lower_img_list = []

lower = open(lower_txt)
for data in lower.readlines():
    data = data[:-1]
    lower_img_list.append(data)
lower.close()

bottle_list = open(bottle_txt)
for line in bottle_list.readlines():
    line = line[:-1]  # 去除换行
    line = line.split()  # path，loc切分
    loc_data = list(map(int, line[1].split(",")[:4]))
    path = line[0]  # 拿到位置
    name = path.split("\\")[-1]  # 位置中的文件名
    temp = name.split(".")[0]  # 去掉后缀的值

    img = cv.imread(img_dir + "/" + name)
    roi = img[loc_data[1]: loc_data[3], loc_data[0]: loc_data[2]]
    if temp in lower_img_list:
        save_path = lower_path + name
    else:
        save_path = upper_path + name
    cv.imwrite(save_path, roi)