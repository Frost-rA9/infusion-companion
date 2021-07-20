import torch


model = torch.load("../Resource/model_data/face_cnn_loss_5.96_acc_1.0.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cuda()


# img_path = "D:/code/Expression/train/3"
img_path = "F:/NutstoreData/code/project_practice/infusion-companion/Resource"
import os
from PIL import Image
import numpy as np
import cv2 as cv
from torchvision import transforms

painful = 0
total = 0

for file in os.listdir(img_path):
    # absolute = img_path + "/" + file
    absolute = img_path + "/" + "face_test2.jpg"
    img = Image.open(absolute)
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (48, 48))
    img = Image.fromarray(img)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img = img.unsqueeze(0)

    output = model(img.to(device))
    output = torch.argmax(output, 1)

    data = output.data.cpu().numpy()[0]
    print(absolute, data)
    if data == 0:
        painful += 1
    total += 1

print(painful, total)
print("painful:", painful / total * 100)
