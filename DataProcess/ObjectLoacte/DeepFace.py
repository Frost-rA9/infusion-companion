# encoding=GBK
"""DeepFace

    -
"""
import cv2 as cv
import matplotlib.pyplot as plt
from deepface import DeepFace

img = cv.imread("../../Resource/face_test.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(img)
# predict = DeepFace.analyze(img)
# print(predict)
faceCascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
# print(cv.data.haarcascades)
# file = open("../../Resource/model_data/haarcascade_frontalface_default.htm")

# detecting face in color_image and getting 4 points(x,y,u,v) around face from the image, and assigning those values to 'faces' variable
faces = faceCascade.detectMultiScale(img, 1.1, 4)
#
# # using that 4 points to draw a rectangle around face in the image
for (x, y, u, v) in faces:
    cv.rectangle(img, (x, y), (x + u, y + v), (0, 0, 225), 2)

plt.imshow(img)
plt.show()
