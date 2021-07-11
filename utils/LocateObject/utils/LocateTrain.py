# encoding=GBK
"""
    - 封装dlib的封装步骤

    - 关于xml文件
        1. imglab.exe -c xxx.xml img_path 生成xml文件
        2. imglab.exe xxx.xml 进行标注
            - 按住shift选定区域
            - 注意在上方的格子里添加label
        3. 完成后file-save
    - 关于svm文件
        1. 通过dlib的训练得到，相当于模型，用于预测
"""

import dlib


class LocateTrain:
    def __init__(self,
                 detect_xml_path: str,
                 output_svm_name: str,
                 open_flips: bool = True,
                 c_count: int = 5,
                 thread_number: int = 8,
                 verbose: bool = True):
        # 设置训练参数和模式
        self.options = dlib.simple_object_detector_training_options()
        # 启用左右翻转
        self.options.add_left_right_image_flips = open_flips
        # 向量机数量
        self.options.C = c_count
        # 线程数量
        self.options.num_threads = thread_number
        # 启动随机森林
        self.options.be_verbose = verbose
        # 启动训练
        output_svm_name = "../../../Resource/svm/trained/" + output_svm_name + ".svm"
        dlib.train_simple_object_detector(detect_xml_path, output_svm_name, self.options)


if __name__ == '__main__':
    # l = LocateTrain("../../../Resource/svm/label/bottle.xml", "bottle_svm", c_count=100)
    from utils.LocateObject.dlibLocate import LocateRoI
    import cv2 as cv
    svm_path = "../../../Resource/svm/trained/bottle_svm.svm"
    # img_path = "H:/pic/liquid/11.png"
    img_path = "F:/DataSet/bottle/Locate/pic/11.png"
    img = cv.imread(img_path)
    L = LocateRoI(svm_path)
    predict_list = L.predict(img)
    for start, end in predict_list:
        cv.rectangle(img, start, end, (0, 255, 255), 2)
    cv.imshow("img", img)
    cv.waitKey(0)



