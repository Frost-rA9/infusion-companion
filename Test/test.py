import cv2 as cv
def get_pic():
    video = cv.VideoCapture(1)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame

# for f in get_pic():
#     cv.imshow("img", f)
#     cv.waitKey(0)
l = [True, False]
if False in l:
    print("hh")
