import torch.nn.functional as F
import torch
import torch.nn as nn
import dsbn

class res_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=0)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d((3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        x = self.conv2(x)
        x += identity
        x = self.relu2(x)
        x = self.maxpool(x)
        return x

class Conv2D_resnet(nn.Module):
    def __init__(self, num_classes, in_ch=2 , p_drop=0.15):
        super(Conv2D_resnet, self).__init__()
        self.conv_layer1 = res_block(in_ch, 32)
        self.conv_layer2 = res_block(32, 64)
        self.conv_layer3 = res_block(64, 128)
        self.conv_layer4 = nn.Conv2d(128, 128, kernel_size=(6, 6), padding=0)
        self.batch0 = nn.BatchNorm1d(128)
        self.relu0 = nn.LeakyReLU()

        self.fc1 = nn.Linear(128, 64)
        self.batch1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=p_drop)
        self.fc2 = nn.Linear(64, 32)
        self.batch2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=p_drop)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv_layer1(x)
        # print(x.size())
        x = self.conv_layer2(x)
        # print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.batch0(x)
        x = self.relu0(x)
        #print(x.size())

        x = self.fc1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
