# encoding=GBK
"""ConvertJson

    - 对于labelme生成的json文件进行转换
    - 转换成(img,label)的形式存储在本地磁盘
"""

import os
import subprocess

d = os.listdir("F:/DataSet/Infusion/Annotations")
main_exe = "D:/Anaconda3/envs/infusion-companion/Scripts/labelme_json_to_dataset.exe"

for address in d:
    address = "F:/DataSet/Infusion/Annotations/" + address
    test = subprocess.Popen(main_exe + " " + address)
    print(test.communicate())

# print(receive)