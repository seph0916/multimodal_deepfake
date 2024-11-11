import torch
import torch.nn as nn
from model.tssd import RSM1D
import torch.nn.functional as F
from torch import flatten

class Multimodal(nn.Module):
    def __init__(self,in_channels=150):
        super(Multimodal, self).__init__()
        # self.num_frames = num_frames
        # self.num_feats = input_len // num_frames
        # self.lstm=nn.LSTM(input_size=self.num_feats, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.01,)
                
        
        self.image_layer_CNN=nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Flatten(),
                                       nn.Linear(128*28*28,4*28*28),
                                       nn.ReLU(), # Relu 넣어봄
                                       nn.Linear(4*28*28,30),
                                       nn.ReLU() # Relu 넣어봄
                                       )
        #6초
        # self.audio_layer_TSSD=nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False),
        #                                     nn.BatchNorm1d(16),
        #                                     nn.ReLU(),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=16, channels_out=32),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=32, channels_out=64),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=64, channels_out=128),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=128,channels_out=128),
        #                                     nn.MaxPool1d(kernel_size=252),
        #                                     nn.Flatten(start_dim=1),
        #                                     nn.Linear(in_features=128,out_features=64),
        #                                     nn.ReLU(),
        #                                     nn.Linear(64,30),
        #                                     nn.ReLU())
        # 10초
        self.audio_layer_TSSD=nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False),
                                            nn.BatchNorm1d(16),
                                            nn.ReLU(),
                                            nn.MaxPool1d(kernel_size=4),
                                            RSM1D(channels_in=16, channels_out=32),
                                            nn.MaxPool1d(kernel_size=4),
                                            RSM1D(channels_in=32, channels_out=64),
                                            nn.MaxPool1d(kernel_size=4),
                                            RSM1D(channels_in=64, channels_out=128),
                                            nn.MaxPool1d(kernel_size=4),
                                            RSM1D(channels_in=128,channels_out=128),
                                            nn.MaxPool1d(kernel_size=1875), # 625 10초 1875 30초
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(in_features=128,out_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64,30),
                                            nn.ReLU())
        
        self.audio_layer_MLP=nn.Sequential(nn.Linear(40*972,120),
                                           nn.ReLU(),
                                           nn.Linear(120, 80),
                                           nn.Sigmoid(),
                                           nn.Linear(80, 30))
        
        self.audio_layer_Shallow=nn.Sequential()
        
        # mfcc, lfcc 전용 conv2d
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(2, 4), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(38144,128) #6초는 7424 10초는 12544 30초는 38144
        self.fc2 = nn.Linear(128, 30)        
        
        
        self.multi_layer=nn.Sequential(nn.Linear(60,1))
        
        self.audio_uni_layer=nn.Sequential(nn.Linear(30,1))
        
        self.image_uni_layer=nn.Sequential(nn.Linear(30,1))
        

        
        
    

    def forward(self, image_model, audio_model, image_input:torch.Tensor, audio_input:torch.Tensor):
        # 이미지 부분 
        if image_model=="CNN":
            image_output=self.image_layer_CNN(image_input) # 콘캣 위한 fc1,2 하나
        
        # 오디오 부분 
        if audio_model=="TSSD": #tssd -> 오디오 부분
            audio_input = audio_input.unsqueeze(dim=1)
            audio_output = self.audio_layer_TSSD(audio_input)
        
        if audio_model=="MLP":
            batch=audio_input.size(0)
            audio_input=audio_input.reshape(batch, -1)
            audio_output=self.audio_layer_MLP(audio_input)
        
        if audio_model=="CNN":
            audio_input=audio_input.unsqueeze(dim=1)
            x = self.pool(F.relu(self.conv1(audio_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = flatten(x, 1)
            x = self.fc1(x)
            audio_output = self.fc2(x)
            
            

        
        
        # 이미지+ 오디오 concat 부분
        multi_input=torch.cat([image_output,audio_output],dim=1)
        multi_output=self.multi_layer(multi_input)
        
        # 이미지, 오디오 각각에 해당 Out put
        image_output=self.image_uni_layer(image_output)
        audio_output=self.audio_uni_layer(audio_output)
        return image_output, audio_output, multi_output


# image_input=torch.rand(16, 30, 224, 224)
# audio_input=torch.rand(16, 2, 96000)
# model=Multimodal()
# a,b,c=model(image_input,audio_input)
# print("aaa")
#random_torch=random_torch.reshape(4,2)
#random_torch.size()
# random_torch1=torch.rand(16,3,224,224)
# random_torch2=torch.rand(16,25*3,224,224)
# torch_stack=torch.cat([random_torch,random_torch1,random_torch2],dim=1) 채널을 콘캣하는 방법.?
# torch_stack.size()
