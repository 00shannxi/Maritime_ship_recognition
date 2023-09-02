
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
from model_cnn_vais_new import Net_227_new,Net_227_ir

import argparse
import torch.optim as optim
from sklearn import metrics
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch Ship Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')

parser.add_argument('--net', default='./model/alexnet.pth', help="path to net (to continue training)")
args = parser.parse_args()


BATCH_SIZE = 32    # 1  4   16   32

if __name__ == '__main__':


    transform = transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    model_class = ('net_395.pth', 'net_400.pth')
    classes = ('medium-other', 'merchant', 'passenger', 'sailing', 'small', 'tug')



    from folderDatasetFushion import FolderDatasetFushion
    from fusionCNN_vais_new import FushionCNN


    testset = tv.datasets.ImageFolder(root='./256_256/ir_256/test/', transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    testset1 = tv.datasets.ImageFolder(root='./256_256/visible_256/test/', transform=transform)
    testloader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False)

    fushionNet=FushionCNN(pretrained=True,modelList=model_class).to(device)


    labels_list = []
    predicted_list = []
    count = 0
    total = 0
    correct = 0
    acc = 0


    weight_1 = 0.6   #0.1-0.9
    weight_2 = 0.4   #0.9-0.1


    for stacked_data_and_label in enumerate(zip(testloader,testloader1)):
        stacked_data1 = stacked_data_and_label[1]
        stacked_data_list = list(stacked_data1)
        stacked_data_ir_and_label, stacked_data_visible_and_label = stacked_data_list
        stacked_data_ir,labels_1_ir = stacked_data_ir_and_label
        stacked_data_visible, labels_1_visible = stacked_data_visible_and_label
        stacked_data = [stacked_data_ir, stacked_data_visible]
        labels_2 = list(labels_1_ir.numpy())

        labels_list.extend(labels_2)

        for i in range(0,len(stacked_data)):
            stacked_data[i]=stacked_data[i].cuda()

        t0 = time.time()
        with torch.no_grad():
            stackedResult_1=fushionNet(stacked_data)
            stackedResult=stackedResult_1.cpu().detach().numpy().swapaxes(0,2)
            result_ir=stackedResult[:, :, 0]
            result_visible = stackedResult[:, :, 1]
            _p=result_ir*weight_1+result_visible*weight_2
            p_result=_p.swapaxes(0,1)
            outputs = torch.Tensor(p_result)

            t1 = time.time()
            print('timer: %.4f sec.' % (t1 - t0))
            with open("test_time_dec.txt", "a") as f6:  # test_time.txt
                f6.write(str('%.4f' % (t1 - t0)) + '\n')

            predicted_1 = outputs.argmax(dim=-1,keepdim=False)
            predicted_list.extend(predicted_1)
            correct += (predicted_1 == labels_1_ir).sum().float()
            total += len(labels_1_ir)
            acc = correct / total

    print("Accuracy:")
    print(acc)
    con_mtx = metrics.confusion_matrix(labels_list, predicted_list)
    m_report=metrics.classification_report(labels_list, predicted_list,digits=4)
    m_precision = metrics.precision_score(labels_list, predicted_list, average="macro")
    m_recall = metrics.recall_score(labels_list, predicted_list, average="macro")
    m_accuracy = metrics.accuracy_score(labels_list, predicted_list)
    f1 = metrics.f1_score(labels_list, predicted_list, average="macro")
    print('confusion_matrix:', con_mtx)
    print('accuracy:{0:.4f}'.format(m_accuracy))
    print('precision:{0:.4f}'.format(m_precision))
    print('recall:{0:0.4f}'.format(m_recall))
    print('f1-score:{0:.8f}'.format(f1))
    print(m_report)




