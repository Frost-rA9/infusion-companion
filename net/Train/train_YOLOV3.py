# -------------------------------------#
#       �����ݼ�����ѵ��
# -------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.YOLOV3.YOLOV3 import YOLOV3 as YoloBody
from net.Train.utils.yolo.yolo_training import YOLOLoss, LossHistory, weights_init
from utils.DataLoader.YoLoLoader import YoloDataset, yolo_dataset_collate


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   �����ݶ�
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   ǰ�򴫲�
            # ----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   ������ʧ
            # ----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            # ----------------------#
            #   ���򴫲�
            # ----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                # ----------------------#
                #   ������ʧ
                # ----------------------#
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


# ----------------------------------------------------#
#   ��⾫��mAP��pr���߼���ο���Ƶ
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # -------------------------------#
    #   �Ƿ�ʹ��Cuda
    #   û��GPU�������ó�False
    # -------------------------------#
    Cuda = True
    # ------------------------------------------------------#
    #   �Ƿ����ʧ���й�һ�������ڸı�loss�Ĵ�С
    #   ���ھ�����������loss�ǳ���batch_size���ǳ�������������
    # ------------------------------------------------------#
    normalize = False
    # ------------------------------------------------------#
    #   �����shape��С
    # ------------------------------------------------------#
    input_shape = (416, 416)
    # ------------------------------------------------------#
    #   ��Ƶ�е�Config.py�Ѿ��Ƴ�
    #   ��Ҫ�޸�num_classesֱ���޸Ĵ˴���num_classes����
    #   �����Ҫ���5����, �����д5. Ĭ��Ϊ20
    # ------------------------------------------------------#
    num_classes = 20
    # ----------------------------------------------------#
    #   �����anchor��·��
    # ----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors = get_anchors(anchors_path)
    # ------------------------------------------------------#
    #   ����yoloģ��
    #   ѵ��ǰһ��Ҫ�޸�Config�����classes����
    # ------------------------------------------------------#
    model = YoloBody(anchors, num_classes)
    weights_init(model)

    # ------------------------------------------------------#
    #   Ȩֵ�ļ��뿴README���ٶ���������
    # ------------------------------------------------------#
    model_path = "model_data/yolo_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # ��������������������������Ե�loss����
    yolo_loss = YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), Cuda, normalize)
    loss_history = LossHistory("logs/")

    # ----------------------------------------------------#
    #   ���ͼƬ·���ͱ�ǩ
    # ----------------------------------------------------#
    annotation_path = '2007_train.txt'
    # ----------------------------------------------------------------------#
    #   ��֤���Ļ�����train.py�����������
    #   2007_test.txt��2007_val.txt����û�������������ġ�ѵ������ʹ�õ���
    #   ��ǰ���ַ�ʽ�£���֤����ѵ�����ı���Ϊ1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   ����������ȡ��������ͨ�ã�����ѵ�����Լӿ�ѵ���ٶ�
    #   Ҳ������ѵ�����ڷ�ֹȨֵ���ƻ���
    #   Init_EpochΪ��ʼ����
    #   Freeze_EpochΪ����ѵ��������
    #   Unfreeze_Epoch��ѵ������
    #   ��ʾOOM�����Դ治�����СBatch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50  # ������Ҫ���ж�����

        optimizer = optim.Adam(net.parameters(), lr)
        # ģ��ѧϰ�ʵ��½���ʽ
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # ------------------------------------#
        #   ����һ������ѵ��
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("���ݼ���С���޷�����ѵ�������������ݼ���")

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        # �ⶳ�����ѵ��
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # ------------------------------------#
        #   �ⶳ��ѵ��
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("���ݼ���С���޷�����ѵ�������������ݼ���")

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()