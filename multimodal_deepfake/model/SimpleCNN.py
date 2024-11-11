import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3 ):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.fc = nn.Linear(64*8*8, 1)  # 하나의 fully connected 층만 사용
        self.fc1 = nn.Linear(64*28*28, 28*28)  # 하나의 fully connected 층만 사용
        self.fc2 = nn.Linear(28*28, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # x = x.view(-1, 64*8*8)
        x = x.view(-1, 64*28*28)
        x = self.fc1(x)  # 하나의 fully connected 층만 사용
        x = self.fc2(x)
        return x.view(-1,1) # 한 영상의 10개의 이미지를 하나의텐서로 바꿔버림
