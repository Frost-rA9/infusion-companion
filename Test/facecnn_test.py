import torch


model = torch.load("../net/Train/5.965041145682335_1.0binary_model.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cuda()


img_path = "D:/code/Expression/train/3"
import os
from PIL import Image
import numpy as np
import cv2 as cv
from torchvision import transforms

painful = 0
total = 0

for file in os.listdir(img_path):
    absolute = img_path + "/" + file

    img = Image.open(absolute)
    img = np.array(img)
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
