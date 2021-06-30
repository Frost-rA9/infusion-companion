# encoding=GBK
"""以块的方式读入内存
    1. 注册数据驱动
    2. 打开栅格数据
    3. 获取波段对象
    4. 读入数据区域
    5. 对数据区域进行滑动窗口显示
"""
import osgeo
from osgeo import gdal
from osgeo.gdalconst import *  # gdal的常量属性
import numpy as np

"""BlockImage
    1. 主要提供图片的地址和块读取范围
    2. 其余的值主要是为了方便
    3. read_data被声明成了可迭代，方便外界调用的程序
"""


class BlockImage:
    def __init__(self, img_path: str,  # 文件路径
                 start_pos: tuple,  # 开始读取的位置
                 end_pos: tuple,  # 结束读取的位置
                 step: tuple,  # 块的读取大小
                 raster_band: osgeo.gdal.Band = None,  # 波段对象
                 driver_name: str = None,  # 驱动名字
                 img_mode: int = None,  # 读取图片的方式
                 ):
        # 1. 导入数据驱动
        if driver_name:
            self.driver = gdal.GetDriverByName(driver_name)
        else:
            self.driver = gdal.GetDriverByName("HFA")
        self.driver.Register()

        # 2. 记录相关信息
        self.img_path = img_path
        self.start_pos = start_pos  # start_x, start_y
        self.end_pos = end_pos  # end_x, end_y
        self.step = step  # step_x, step_y

        if raster_band:  # 如果不是空的话就对这一个波段读取
            self.raster_band = raster_band
        else:
            self.raster_band = None

        if img_mode:  # 图像的读取方式
            self.img_mode = img_mode
        else:
            self.img_mode = GA_ReadOnly

        # 3. 文件打开
        self.img_opened = gdal.Open(self.img_path, self.img_mode)

    def set_info(self, start_pos: tuple = None,
                 end_pos: tuple = None,
                 step: tuple = None):
        """可以在后续中用同一个对象读取不同位置的信息"""
        if start_pos:
            self.start_pos = start_pos
        if end_pos:
            self.end_pos = end_pos
        if step:
            self.step = step

    def read_data(self):
        """搞成一个可迭代函数，让调用的人方便处理"""
        start_x, end_x, step_x = self.start_pos[0], self.end_pos[0], self.step[0]
        start_y, end_y, step_y = self.start_pos[1], self.end_pos[1], self.step[1]
        for row in range(start_y, end_y, step_y):
            if row + step_y < end_y:
                num_rows = step_y
            else:
                num_rows = step_y - row

            for col in range(start_x, end_x, step_x):
                if col + step_x < end_x:
                    num_cols = step_x
                else:
                    num_cols = step_x - col

                yield self.read_band((col, row), (num_cols, num_rows))

    def read_band(self, start_pos: tuple, step: tuple):
        """
        :param start_pos: (start_x, start_y)
        :param step: (step_x, step_y)
        :return: band.ReadAsArray
        """
        start_x, start_y = start_pos
        step_x, step_y = step
        if self.raster_band:
            return self.raster_band.ReadAsArray(start_x, start_y, step_x, step_y)

        bands = self.img_opened.RasterCount  # 获取所有的波段
        img_list = []
        for band in range(1, bands + 1):
            _band = self.img_opened.GetRasterBand(band)
            data = _band.ReadAsArray(start_x, start_y, step_x, step_y)
            img_list.append(data)
            # print(img_list[-1])
        return np.stack(img_list, axis=2)


if __name__ == '__main__':
    """示例代码"""

    img_path = "../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif"
    big_img = "F:/DataSet/A3/E117D5_N34D2_20180204_GF2_DOM_4_fus/E117D5_N34D2_20180204_GF2_DOM_4_fus.tif"
    start, end, step = (20000, 20000), (100000, 100000), (200, 200)
    Block = BlockImage(big_img, start, end, step)
    import cv2 as cv
    np.set_printoptions(threshold=np.inf)
    for img in Block.read_data():
        print(img)
        cv.imshow("img", img)
        cv.waitKey(0)
