# encoding=GBK
"""CSVDecoder

    - 文件格式

    emotion: 表情类别
    pixels: 灰度图
    Usage: 用途

    - 晚点写
"""
from PIL import Image
import pandas as pd
import cv2 as cv
import numpy as np
import os
train_path="D:/DataSet/CAER-S/test"
test_path="D:/DataSet/CAER-S/train"
file_path = "../../Resource/fer2013.csv"



def make_dir():
    for i in range(0,8):
        p1 = os.path.join(train_path,str(i))
        p2 = os.path.join(test_path,str(i))
        if not os.path.exists(p1):
            os.makedirs(p1)
        if not os.path.exists(p2):
           os.makedirs(p2)

def save_images():
    df = pd.read_csv(file_path)
    t_i = [1 for i in range(0,7)]
    v_i = [1 for i in range(0,7)]
    for index in range(len(df)):

        emotion = df.loc[index][0]
        usage = df.loc[index][2]
        image = df.loc[index][1]

        img = np.array(image.split()).astype(np.uint8)
        img = img.reshape((48, 48))


        if(usage=='Training'):
            t_p = os.path.join(train_path,str(emotion),'{}.jpg'.format(t_i[emotion]))
            cv.imwrite(t_p, img)
            t_i[emotion] += 1

        else:
            v_p = os.path.join(test_path,str(emotion),'{}.jpg'.format(v_i[emotion]))
            cv.imwrite(v_p, img)
            v_i[emotion] += 1


make_dir()
save_images()

