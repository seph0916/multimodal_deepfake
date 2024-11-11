import torch
import torch.nn as nn

class fclayer(nn.Module):
    def __init__(self):
        super(fclayer, self).__init__()
        self.fc1 = nn.Linear(50, 1)  # fc2레이어로 돌진.


    def forward(self, x:torch.Tensor):
        x= x.view(1,-1)
        x = self.fc1(x)
        return x.view(-1,1)



class unilayer(nn.Module):
    def __init__(self):
        super(unilayer, self).__init__()
        self.fc1 = nn.Linear(25, 1)  # fc2레이어로 돌진.


    def forward(self, x:torch.Tensor):
        x= x.view(1,-1)
        x = self.fc1(x)
        return x.view(-1,1)
