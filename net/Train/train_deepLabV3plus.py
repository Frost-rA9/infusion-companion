# encoding=GBK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from net.DeepLabV3Plus.DeepLabV3Plus import DeepLabV3Plus
from net.DeepLabV3Plus.Classifier.Classifier import Classifier
from net.BoneNet.ResNet.ResNet101 import ResNet101
from utils.DataLoader.DeepLabSingleLoader import DeepLabSingleLoader

import numpy as np
torch.set_printoptions(threshold=np.inf)

# 1. ģ�ͽ���
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet101.get_res_net101(BatchNorm=torch.nn.BatchNorm2d, pretrained=True, output_stride=8,
                                 pretrained_loc="../../Resource/model_data/resnet101-5d3b4d8f.pth")
classifier = Classifier(middle_channels=256, out_channels=2048, num_classes=3)
deep_lab_v3_plus = DeepLabV3Plus(model, classifier)
deep_lab_v3_plus.to(device)

# 2. ��ʧ���Ż�������
criterion = nn.BCELoss()
optimizer = optim.Adam(deep_lab_v3_plus.parameters(), lr=0.0001)

# 3. ���ݼ�ת��
train_trans = transforms.Compose([
    transforms.Resize((400, 300)),
    transforms.ToTensor(),
])

test_trans = transforms.Compose([
    transforms.Resize((400, 300)),
    transforms.ToTensor(),
])

data_train = DeepLabSingleLoader("F:/DataSet/bottle/segmentation/dir_json/train/", train_trans,
                                 test_trans, num_classes=3, is_train=True)
data_test = DeepLabSingleLoader("F:/DataSet/bottle/segmentation/dir_json/test/", train_trans,
                                test_trans, num_classes=3, is_train=False)
data_train_loader = DataLoader(dataset=data_train, batch_size=1, shuffle=True)
data_test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=False)


def train():
    deep_lab_v3_plus.train()
    total_loss = 0
    train_size = 1
    count = 0
    for epoc in range(10000):
        for idx, (image, label) in enumerate(data_train_loader):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = deep_lab_v3_plus(image)
            output = nn.Softmax(dim=1)(output)

            loss = criterion(output, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if count % train_size == train_size - 1:
                print("[epoc:%3d, idx:%3d] totoal_loss: %.3f" % (epoc + 1, count + 1, total_loss))
                test(total_loss)
                total_loss = 0.0
            count += 1


def test(loss: float):
    deep_lab_v3_plus.eval()
    total_correct = 0
    total = 0
    threshold_accuracy = 0.95

    # ����Ҫ�����ݶȣ��������ѵ���ٶ�
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1)  # ȥ��ͨ����ӦΪ������
            output = deep_lab_v3_plus(images)
            output = nn.Softmax(dim=1)(output)
            output = torch.argmax(output, dim=1)  # �ҳ�ÿ�����

            total_correct += output.eq(labels).int().sum().item()  # ͳ�ƺͱ�ǩһ���ĸ���
            total += labels.size(1) * labels.size(2)  # width * height

            current_accuracy = total_correct / total
            print("current_accuracy", current_accuracy)
            if current_accuracy > threshold_accuracy:
                print("mode current_accuracy is :", current_accuracy)
                torch.save(deep_lab_v3_plus,
                           "./loss_" + str(loss) + "_" + str(current_accuracy)[:6] + "_.pth")


if __name__ == '__main__':
    train()
