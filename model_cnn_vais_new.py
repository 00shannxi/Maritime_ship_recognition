

import torch
import torch.nn.functional as F

class Net_227_new(torch.nn.Module):
    def __init__(self):
        super(Net_227_new, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 192, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 384, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 2048),
            #torch.nn.Linear(4096, 4)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 6)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        # conv5_out = self.conv5(conv4_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out

class Net_227_ir(torch.nn.Module):
    def __init__(self):
        super(Net_227_ir, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 192, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 384, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 2048),
            #torch.nn.Linear(4096, 4)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 6)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        # conv5_out = self.conv5(conv4_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


if __name__=="__main__":
    print(Net_227_ir())
    print(Net_227_new())
