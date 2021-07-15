# import cv2 as cv
# def get_pic():
#     video = cv.VideoCapture(1)
#
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         yield frame
#
# # for f in get_pic():
# #     cv.imshow("img", f)
# #     cv.waitKey(0)
# l = [True, True]
# if False in l:
#     print("hh")
# l = [0] * 3
# # l[1] = 4
# print(max(l))
# import numpy as np
# import cv2 as cv
# array = np.array([1, 2, 3])
# l = []
# if l is []:
#     print(True)
# else:
#     print(False)
# print(len(array))
# print(len(l))
#
# mask = np.zeros((200, 200), np.uint8)
# ret, mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)
# cv.imshow("mask", mask)
# cv.waitKey(0)
l = [1, 2]
print(l[:2])
