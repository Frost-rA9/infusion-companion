import cv2


class number:
    def __init__(self):
        self.index = 0
        self.count()

    def count(self):
        while True:
            # cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                break
            else:
                self.index += 1
            cap.release()
