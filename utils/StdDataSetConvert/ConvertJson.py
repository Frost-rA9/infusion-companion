# encoding=GBK
"""ConvertJson

    - 对于labelme生成的json文件进行转换
    - 转换成(img,label)的形式存储在本地磁盘
"""

import os
import subprocess

data_dir = "H:/boo_des"

d = os.listdir(data_dir)
main_exe = "D:/Anaconda3/envs/infusion-companion/Scripts/labelme_json_to_dataset.exe"

for address in d:
    address = data_dir + "/" + address
    test = subprocess.Popen(main_exe + " " + address)
    print(test.communicate())

# print(receive)