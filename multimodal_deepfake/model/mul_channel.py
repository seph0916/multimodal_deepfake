import torch
import torch.nn as nn
from model.tssd import RSM1D

class Multimodal(nn.Module):
    def __init__(self, in_channels=30, num_frames=10, input_len=96000, hidden_dim=30, out_dim=30 ):
        super(Multimodal, self).__init__()
        self.num_frames = num_frames
        self.num_feats = input_len // num_frames
        self.lstm=nn.LSTM(input_size=self.num_feats, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.01,)
        
        
        self.image_layer=nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
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
                                       #nn.Linear(28*28,25)  # ReLU따로 안쓰는지? 이후확인.
                                       )
        self.audio_layer=nn.Sequential(nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       nn.Linear(hidden_dim * 2 * num_frames, out_dim))
        
        self.audio_tssd_layer=nn.Sequential( nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, padding=3, bias=False),
                                             nn.BatchNorm1d(16), 
                                             nn.ReLU(),
                                             nn.MaxPool1d(kernel_size=4),
                                             RSM1D(channels_in=16, channels_out=32),
                                             nn.MaxPool1d(kernel_size=4),
                                             RSM1D(channels_in=32, channels_out=64),
                                             nn.MaxPool1d(kernel_size=4),
                                             RSM1D(channels_in=64, channels_out=128),
                                             nn.MaxPool1d(kernel_size=4),
                                             RSM1D(channels_in=128, channels_out=128),
                                             nn.MaxPool1d(kernel_size=4),
                                             nn.Flatten(),
                                             nn.Linear(in_features=128, out_features=64),
                                             nn.ReLU(),
                                             nn.Linear(64,30))
        
        self.multi_layer=nn.Sequential(nn.Linear(60,1))
        
        self.audio_uni_layer=nn.Sequential(nn.Linear(30,1))
        
        self.image_uni_layer=nn.Sequential(nn.Linear(30,1))
        


        
    

    def forward(self, image_input:torch.Tensor, audio_input:torch.Tensor):
        # 이미지 부분 
        image_output=self.image_layer(image_input) # 콘캣 위한 fc1,2 하나
        audio_output=self.audio_tssd_layer(audio_input)
        # # lstm -> 오디오 부분
        # batch = audio_input.size(0)
        # audio_input = audio_input.reshape(batch, self.num_frames, self.num_feats)
        # lstm_out, _ = self.lstm(audio_input)  # (B, T, C=hidden_dim * 2)
        # feat = lstm_out.permute(0, 2, 1)  # (B, C=hidden_dim * 2, T)
        # audio_output=self.audio_layer(feat) # 콘캣 위한 fc1 하나
        
        # #tssd -> 오디오 부분
        # x = audio_input.unsqueeze(1)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=4)

        # # stacked ResNet-Style Modules
        # x = self.RSM1(x)
        # x = F.max_pool1d(x, kernel_size=4)
        # x = self.RSM2(x)
        # x = F.max_pool1d(x, kernel_size=4)
        # x = self.RSM3(x)
        # x = F.max_pool1d(x, kernel_size=4)
        # x = self.RSM4(x)
        # x = F.max_pool1d(x, kernel_size=x.shape[-1])

        # x = torch.flatten(x, start_dim=1)
        # x = F.relu(self.fc1(x))
        # audio_output = F.relu(self.fc2(x))
        

                
        # 이미지+ 오디오 concat 부분
        multi_input=torch.cat([image_output,audio_output],dim=1)
        multi_output=self.multi_layer(multi_input)
        
        # 이미지, 오디오 각각에 해당 Out put
        image_output=self.image_uni_layer(image_output)
        audio_output=self.audio_uni_layer(audio_output)
        return image_output, audio_output, multi_output



image_input=torch.rand(16,30,224,224)
audio_input=torch.rand(2,96000)
a,b,c=Multimodal(image_input,audio_input)

