# encoding=GBK
"""ConvertJson

    - ����labelme���ɵ�json�ļ�����ת��
    - ת����(img,label)����ʽ�洢�ڱ��ش���
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