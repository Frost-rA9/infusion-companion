# encoding=GBK
"""VOCPathChange

    - ת����ע�õ�xml���ݵ�ʵ���������

    <folder>ͼƬ</folder>
	<filename>1.png</filename>
	<path>C:/Users/33198/Desktop/ͼƬ/1.png</path>

    1. ��folder�е�Ŀ¼�����Ϊ��ǰ�������ڵ�Ŀ¼��
    2. ��path�е����ݽ��и���(�򵥵�replace)
"""

import os

xml_path = "F:/DataSet/bottle/Locate/Annotations"
d = os.listdir(xml_path)

for file_name in d:
    file_path = xml_path + "/" + file_name
    # file_path = "F:/DataSet/bottle/Locate/Annotations/1.xml"
    file = open(file_path, "r", encoding="utf-8")
    document = file.read()
    document = document.replace("<folder>ͼƬ</folder>", "<folder>pic</folder>")
    document = document.replace("C:\\Users\\33198\\Desktop\\ͼƬ", "F:\\DataSet\\bottle\\Locate\\pic")
    file.close()

    file = open(file_path, "w", encoding="utf-8")
    file.write(document)
    file.close()
