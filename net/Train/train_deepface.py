import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



# path1=os.path.join("C:\\Users\\33198\Desktop\\1.jpg")
# path2 = os.path.join("C:\\Users\\33198\Desktop\\2.jpg")
# image1 = cv2.imread(path1)
#
#
# plt.imshow(image1[:, :, ::-1])
# plt.show()
#
# image2 = mpimg.imread(path2)
# plt.imshow(image2)
# plt.show()
#
# from deepface import DeepFace
# result= DeepFace.verify(path1,path2)
# print("Is verified:",result["verified"])
# print(DeepFace.analyze(image1))
from pathlib import Path
home = str(Path.home())
output = home+'/.deepface/weights/vgg_face_weights.h5'
print(output)