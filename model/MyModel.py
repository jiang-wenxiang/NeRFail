import torch
from torch import nn
from torch.nn import functional as F

class MyCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.max_pool5 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(256, 128, 3)
        self.max_pool6 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(128, 64, 3)
        self.max_pool7 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool6(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.max_pool7(x)
        x = x.reshape(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x

