# encoding=GBK
"""对SlidingBlockImage的优化
    - 主要是增加了无限循环的部分
"""


from utils.ImageRead.Sliding import SlidingBlockImage


class ImageBlockHelper(SlidingBlockImage):
    def __init__(self, img_path, step: tuple):
        super().__init__(img_path, None, None, None)
        self.img_path = img_path
        self.step = step
        end_x, end_y, _ = self.read_base_info()
        super().__init__(self.img_path, (0, 0),
                         (end_x, end_y), self.step)

    def read_data_loop(self):
        while True:
            for data in self.read_data():
                yield data[0]
            # print("read finish")


if __name__ == '__main__':
    b = ImageBlockHelper("../../../../Resource/GF2_PMS1__L1A0001064454-MSS1.tif",
                         step=(224, 224))
    import cv2 as cv
    import numpy as np
    for data in b.read_data_loop():
        data = data.astype(np.uint8)
        cv.imshow("data", data)
        cv.waitKey(0)

