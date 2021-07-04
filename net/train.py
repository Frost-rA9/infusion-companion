# encoding=GBK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from utils.DataLoader.LoadSingleFile import LoadSingleFile
from GoogLeNetV4.Googlenetv4 import Googlenetv4
from utils.ImageExpand.ExpandMethod import ExpandMethod

"""=============ȫ�ֲ�������Ҫ�ǵ��η���=========="""

best_accuracy = 0.65
train_size = 40


"""==============������ѵ��=================="""


def train(data_train_loader: DataLoader, data_test_loader: DataLoader):
    net.train()
    total_loss = 0
    global train_size
    count = 0
    for epoc in range(10000):
        for idx, (image, label) in enumerate(data_train_loader):
            image, label = image.to(device), label.to(device)
            print(image.shape, label)

            optimizer.zero_grad()
            output = net(image)

            loss = criterion(output, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if count % train_size == train_size - 1:
                # ÿtrain_sizeһ��
                print("[epoc:%3d, idx:%3d] loss: %.3f" % (epoc + 1, count + 1, total_loss / train_size))
                total_loss = 0.0
                test(data_test_loader)
            count += 1


def test(data_test_loader: DataLoader):
    net.eval()
    total_correct = 0
    total = 0
    global best_accuracy
    global train_size

    # ����Ҫ�����ݶȣ��������ѵ���ٶ�
    with torch.no_grad():
        # ��ѵ��������һ�㣬���Լ�����һ��
        for idx, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)

            output = net(images)
            output = torch.argmax(output, dim=1)  # �ҳ�ÿһ�е����ֵ
            print(output, labels)
            total_correct += output.eq(labels.view_as(output)).sum().item()  # ͳ�ƺͱ�ǩһ���ĸ���
            total += labels.size(0)

            current_accuracy = total_correct / total
            print("current_accuracy", current_accuracy, "best_accruacy", best_accuracy)
            if current_accuracy > best_accuracy:
                print("mode current_accuracy is :", current_accuracy)
                torch.save(net, "./" + str(current_accuracy)[:6] + "_model.pth")


if __name__ == '__main__':
    """ģ��ѵ��"""
    expand_method = ExpandMethod()
    method = expand_method.get_transform()
    # train_trans = transforms.Compose([transforms.ToTensor(), ])
    train_set = LoadSingleFile(train_path="E:/DataSet/CAER-S/test",
                               test_path="E:/DataSet/CAER-S/train",
                               is_train=True,
                               trans=method)
    test_trans = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((400, 536)),
        transforms.ToTensor(),
    ])
    test_set = LoadSingleFile(train_path="E:/DataSet/CAER-S/test",
                               test_path="E:/DataSet/CAER-S/train",
                               is_train=False,
                               trans=test_trans)

    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=6, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Googlenetv4(num_classes=train_set.get_num_classes())
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)

    train(train_loader, test_loader)
