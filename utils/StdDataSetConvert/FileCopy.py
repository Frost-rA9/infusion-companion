# encoding=GBK
"""主要是为了批量修改文件名并且把文件复制到对应位置
"""

start_loc = "H:/bottle"
des_loc = "F:/DataSet/bottle/Locate/JPEGImages"
start_count = 298

import os
import shutil
import sys
for file in os.listdir(start_loc):
    start = start_loc + "/" + file
    des = des_loc + "/" + str(start_count) + ".jpg"
    start_count += 1
    try:
        shutil.copy(start, des)
    except IOError as e:
        print("unable to copy file", e)
    except:
        print("unexpected error", sys.exc_info())