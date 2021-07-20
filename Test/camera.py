import cv2 as cv
import os
targetFile=''
capture = cv.VideoCapture(0)
filepath='D:/人脸数据集/bottle'
os.chdir(filepath)
count = 1120

while True:
    ret, frame = capture.read()
    if not ret:
        break

    cv.imshow("img", frame)
    k=cv.waitKey(27)
    if k == 27:  # 按下esc时，退出
        exit(0)

    if k == ord('d'):
        count -= 1
        targetFile = str(count) + ".jpg"
        os.remove(filepath+'/'+targetFile)
        print("已删除")
    elif k == ord('s'):  # 按下s键时保存并退出
        frame = cv.resize(frame, (800, 600))
        cv.imwrite(str(count) + ".jpg", frame)
        count += 1
        print("保存好了")