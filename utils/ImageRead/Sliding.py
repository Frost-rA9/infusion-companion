# encoding=GBK
"""�������ڵ�ʵ��
    - ������������ʾ��������ٻ������ڵ����ݱ䶯
        1. �������ڵ��ڴ���̭���ԣ��ͷŻ��棩
        2. �������ڵ����ݱ䶯���ԣ����뻺�棩
    - �����ӵ㴦��
        1. ��ʵ���Ǵ��ڵ����ĵ���뿪ʼλ�õ�ƫ����
        2. Ȼ��������봰�ڴ�С������
    - ��������
        1. ���ڱ����ߵ����Ų���
        2. ʵ���Ͼ�������ȡ�㣬���������ܶ�
            - ��Ȼ��������������������ڴ���ȶ���
            - ͬʱ����˼����ٶ�
    - ����ÿ��ģ��
        1. ϣ����ģ�����̵߳�������Ϊ�ܺ�ʱ��
        2. ��Ҫ�Ǽ��ٳ��򿨶ٵ����
"""

from utils.ImageRead.Block import BlockImage
import numpy as np
import warnings
import threading

"""������SlidingBlockImage
    - ��Ҫ�Ƕ�ԭ����BlockImage���й�������
        1. ��ȡͼ��Ļ�����Ϣ���ڶԻ������ڵ�λ�ý���У׼
    - ���Բ�����Ϣ������д������Ӧ��������
        1. ��д��read_data�ṩ��ÿ����Ⱦ��λ��
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
        """���һ���ɵ����������õ��õ��˷��㴦��"""
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
    """������,���ڻ������ڵ���Ⱦ"""

    def __init__(self, sliding_helper: SlidingBlockImage,
                 win_graph: np.ndarray,
                 win_offset: list):
        super().__init__()
        self.sliding_helper = sliding_helper
        self.win_graph = win_graph
        self.win_offset = win_offset

    def render(self):
        """��Ⱦ����"""
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
        # 1. ��ʼ��������Ϣ
        self.win_size = win_size  # ���ڴ�С(width, height)
        self.win_graph = np.zeros((self.win_size[1], self.win_size[0], 3),
                                  dtype=np.uint8)  # �����ʹ���һ����С�Ĵ���Ⱦͼ
        self.img_file = img_file  # ����һ���ļ�·��
        self.win_offset = []  # ͼ���ƫ��λ��

        # 2. �������������
        self.sliding_helper = SlidingBlockImage(self.img_file, None, None, None)
        self.init_helper()

    """1. ��ʼ��������"""
    def init_helper(self):
        cols, rows, _ = self.sliding_helper.read_base_info()
        start_pos = (cols // 2, rows // 2)  # ͼ������
        self.win_offset = [cols // 2, rows // 2]  # ֮�󴰿ڵĻ����Ե���offsetΪ׼

        end_pos = (start_pos[0] + self.win_size[0],
                   start_pos[1] + self.win_size[1])
        step = (self.win_size[0] // 4, self.win_size[1] // 4)
        self.sliding_helper.set_info(start_pos, end_pos, step)  # ϣ����16�˶�ȡ����ɲ���

    """�����ࣺ���߳�ͼ�񻬿���Ⱦ"""
    def render(self):
        render_thread = Render(self.sliding_helper,
                               self.win_graph,
                               self.win_offset)
        render_thread.start()

    """2. ����ƫ����, ���ڴ���"""
    def set_offset(self, x_offset, y_offset):
        self.win_offset[0] += x_offset
        self.win_offset[1] += y_offset

    def reset_helper(self, step_: int = 4):
        start_pos = self.win_offset[0], self.win_offset[1]
        end_pos = (start_pos[0] + self.win_size[0],
                   start_pos[1] + self.win_size[1])
        step = (self.win_size[0] // step_, self.win_size[1] // step_)
        self.sliding_helper.set_info(start_pos, end_pos, step)

    """ . ������������ͼ���ȡ"""
    def show_win_graph(self):
        self.render()
        while True:
            yield self.win_graph


if __name__ == '__main__':
    """ʾ������"""
    import cv2 as cv

    np.set_printoptions(threshold=np.inf)
    img_path = "../../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif"
    s = Sliding((700, 700), img_path)
    for graph in s.show_win_graph():
        cv.imshow("graph", graph)
        cv.waitKey(0)
