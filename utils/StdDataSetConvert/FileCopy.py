# encoding=GBK
"""��Ҫ��Ϊ�������޸��ļ������Ұ��ļ����Ƶ���Ӧλ��
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