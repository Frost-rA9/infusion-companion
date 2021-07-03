# encoding=GBK
"""CSVDecoder

    - 文件格式

    emotion: 表情类别
    pixels: 灰度图
    Usage: 用途

    - 晚点写
"""

import pandas as pd
import cv2 as cv
import numpy as np

file_path = "../../Resource/fer2013.csv"
df = pd.read_csv(file_path)
img = np.array(df.iloc[0][1].split()).astype(np.uint8)
img = img.reshape((48, 48))
cv.imshow("img", img)
cv.waitKey(0)