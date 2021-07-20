# encoding=GBK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import random
from PIL import Image
from utils.DataLoader.LoadSingleFile import LoadSingleFile
from net.GoogLeNetV4.Googlenetv4 import Googlenetv4
from utils.ImageExpand.ExpandMethod import ExpandMethod
from utils.Caculate.ImageFolder import ImageFolder

# torch.set_printoptions(threshold=np.inf)
"""=============全局参量，主要是调参方便=========="""
best_accuracy = 0.70
train_size = 200

"""==============神经网络训练=================="""


def train(data_train_loader: DataLoader, data_test_loader: DataLoader):
    net.train()
    total_loss = 0
    global train_size
    for epoc in range(10000):
        for idx, (image, label) in enumerate(data_train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, label)

            # print("label", label, "loss", loss)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            if idx % train_size == train_size - 1:
                # 每train_size一轮
                print("[epoc:%3d, idx:%3d] loss: %.3f" % (epoc + 1, idx + 1, total_loss))

                test(data_test_loader,total_loss)
                total_loss = 0.0


def test(data_test_loader: DataLoader, loss):
    net.eval()
    total_correct = 0
    total = 0
    global best_accuracy
    global train_size

    # count = 200#random.randint(200, data_test_loader.__len__() //1)
    # 不需要计算梯度，进而提高训练速度
    with torch.no_grad():
        # 让训练集多跑一点，测试集减少一点
        for idx, (images, labels) in enumerate(data_test_loader):
            # print(idx)
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            output = torch.argmax(output, dim=1)  # 找出每一行的最大值
            # print("output", output, "labels", labels)
            total_correct += output.eq(labels).int().sum().item()  # 统计和标签一样的个数
            total += labels.size(0)

            current_accuracy = total_correct / total

            if current_accuracy > best_accuracy:
                print("mode current_accuracy is :", current_accuracy)
                torch.save(net, "./" + str(loss) + "_" + str(current_accuracy)[:6] + "_model.pth")

            # if total > count:
            #     break
        else:
            print("current_accuracy", current_accuracy, "best_accruacy", best_accuracy)

if __name__ == '__main__':
    """模型训练"""
    expand_method = ExpandMethod()
    train_trans = expand_method.get_transform()
    test_trans = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Grayscale(3),
        # transforms.Resize((400, 400)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5077325, 0.5077325, 0.5077325), std=(0.21166028, 0.21166028, 0.21166028)),
    ])

    # train_trans = transforms.Compose([transforms.ToTensor(), ])
    train_set = LoadSingleFile(train_path="D:/code/Expression/train",
                               test_path="D:/code/Expression/test",
                               is_train=True,
                               trans=train_trans,
                               resize=True)

    test_set = LoadSingleFile( train_path="D:/code/Expression/train",
                               test_path="D:/code/Expression/test",
                               is_train=False,
                               trans=test_trans,
                               resize=True)
    # img,lab=test_set.__getitem__(1)
    # print(img,lab)
    # image = img.cpu().clone()
    # image = image.squeeze(0)
    # image = unloader(image)
    # print(img)
    # exit(0)

    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=6, shuffle=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Googlenetv4(num_classes=train_set.get_num_classes())

    pretrained_path = "263.0037250816822_0.75_model.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dict = net.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print('Finished!')

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.1)
    #momentum 当本次梯度下降- dx * lr的方向与上次更新量v的方向相同时，上次的更新量能够对本次的搜索起到一个正向加速的作用

    train(train_loader, test_loader)
    # test(test_loader)