# encoding=GBK
"""滑动窗口的实现
    - 大量的像素显示，必须减少滑动窗口的数据变动
        1. 滑动窗口的内存淘汰策略（释放缓存）
        2. 滑动窗口的数据变动策略（读入缓存）
    - 关于视点处理
        1. 其实就是窗口的中心点距离开始位置的偏移量
        2. 然后就是载入窗口大小的数据
    - 关于缩放
        1. 基于比例尺的缩放策略
        2. 实际上就是跳着取点，减少像素密度
            - 虽然降低质量，但是提高了内存的稳定性
            - 同时提高了加载速度
    - 关于每个模块
        1. 希望把模块变成线程的任务，因为很耗时间
        2. 主要是减少程序卡顿的情况
"""

from utils.ImageRead.Block import BlockImage
import numpy as np
import warnings
import threading

"""辅助类SlidingBlockImage
    - 主要是对原来的BlockImage进行功能扩充
        1. 读取图像的基本信息用于对滑动窗口的位置进行校准
    - 并对部分信息进行重写，以适应滑动窗口
        1. 重写了read_data提供了每次渲染的位置
"""


class SlidingBlockImage(BlockImage):
    def __init__(self, img_path, start_pos, end_pos, step):
        super().__init__(img_path, start_pos, end_pos, step)

    def read_base_info(self):
        cols = self.img_opened.RasterXSize
        rows = self.img_opened.RasterYSize
        bands = self.img_opened.RasterCount
        return cols, rows, bands

    def read_data(self):
        """搞成一个可迭代函数，让调用的人方便处理"""
        start_x, end_x, step_x = self.start_pos[0], self.end_pos[0], self.step[0]
        start_y, end_y, step_y = self.start_pos[1], self.end_pos[1], self.step[1]
        for row in range(start_y, end_y, step_y):
            if row + step_y < end_y:
                num_rows = step_y
            else:
                num_rows = end_y - row

            for col in range(start_x, end_x, step_x):
                if col + step_x < end_x:
                    num_cols = step_x
                else:
                    num_cols = end_x - col
                img_data = self.read_band((col, row), (num_cols, num_rows))
                loc_data = col, row, num_cols, num_rows
                yield img_data, loc_data


class Render(threading.Thread):
    """辅助类,用于滑动窗口的渲染"""

    def __init__(self, sliding_helper: SlidingBlockImage,
                 win_graph: np.ndarray,
                 win_offset: list):
        super().__init__()
        self.sliding_helper = sliding_helper
        self.win_graph = win_graph
        self.win_offset = win_offset

    def render(self):
        """渲染函数"""
        for img_data, loc_data in self.sliding_helper.read_data():
            col, row, num_cols, num_rows = loc_data
            col -= self.win_offset[0]
            row -= self.win_offset[1]
            self.win_graph[row: row + num_rows, col: col + num_cols] = img_data

    def run(self):
        self.render()


class Sliding:
    def __init__(self, win_size: tuple,
                 img_file: str):
        # 1. 初始化基础信息
        self.win_size = win_size  # 窗口大小(width, height)
        self.win_graph = np.zeros((self.win_size[1], self.win_size[0], 3),
                                  dtype=np.uint8)  # 创建和窗口一样大小的待渲染图
        self.img_file = img_file  # 这是一个文件路径
        self.win_offset = []  # 图像的偏移位置

        # 2. 创建辅助类对象
        self.sliding_helper = SlidingBlockImage(self.img_file, None, None, None)
        self.init_helper()

    """1. 初始化辅助类"""
    def init_helper(self):
        cols, rows, _ = self.sliding_helper.read_base_info()
        start_pos = (cols // 2, rows // 2)  # 图像中心
        self.win_offset = [cols // 2, rows // 2]  # 之后窗口的滑动以调整offset为准

        end_pos = (start_pos[0] + self.win_size[0],
                   start_pos[1] + self.win_size[1])
        step = (self.win_size[0] // 4, self.win_size[1] // 4)
        self.sliding_helper.set_info(start_pos, end_pos, step)  # 希望在16此读取内完成操作

    """辅助类：多线程图像滑块渲染"""
    def render(self):
        render_thread = Render(self.sliding_helper,
                               self.win_graph,
                               self.win_offset)
        render_thread.start()

    """2. 设置偏移量, 用于窗口"""
    def set_offset(self, x_offset, y_offset):
        self.win_offset[0] += x_offset
        self.win_offset[1] += y_offset

    def reset_helper(self, step_: int = 4):
        start_pos = self.win_offset[0], self.win_offset[1]
        end_pos = (start_pos[0] + self.win_size[0],
                   start_pos[1] + self.win_size[1])
        step = (self.win_size[0] // step_, self.win_size[1] // step_)
        self.sliding_helper.set_info(start_pos, end_pos, step)

    """ . 主函数，用于图像读取"""
    def show_win_graph(self):
        self.render()
        while True:
            yield self.win_graph


if __name__ == '__main__':
    """示例代码"""
    import cv2 as cv

    np.set_printoptions(threshold=np.inf)
    img_path = "../../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif"
    s = Sliding((700, 700), img_path)
    for graph in s.show_win_graph():
        cv.imshow("graph", graph)
        cv.waitKey(0)
