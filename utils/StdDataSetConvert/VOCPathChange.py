# encoding=GBK
"""VOCPathChange

    - 转换标注好的xml内容到实体机的内容

    <folder>图片</folder>
	<filename>1.png</filename>
	<path>C:/Users/33198/Desktop/图片/1.png</path>

    1. 将folder中的目录名变更为当前数据所在的目录名
    2. 将path中的内容进行跟换(简单的replace)
"""

import os

xml_path = "F:/DataSet/bottle/Locate/Annotations"
d = os.listdir(xml_path)

for file_name in d:
    file_path = xml_path + "/" + file_name
    # file_path = "F:/DataSet/bottle/Locate/Annotations/1.xml"
    file = open(file_path, "r", encoding="utf-8")
    document = file.read()
    document = document.replace("<folder>图片</folder>", "<folder>pic</folder>")
    document = document.replace("C:\\Users\\33198\\Desktop\\图片", "F:\\DataSet\\bottle\\Locate\\pic")
    file.close()

    file = open(file_path, "w", encoding="utf-8")
    file.write(document)
    file.close()
