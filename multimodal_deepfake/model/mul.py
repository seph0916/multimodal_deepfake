import torch
import torch.nn as nn
from model.tssd import RSM1D
import torch.nn.functional as F
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
        self.audio_tssd_layer=nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False),
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
                                            nn.MaxPool1d(kernel_size=252),
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(in_features=128,out_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64,30),
                                            nn.ReLU())
        
        self.multi_layer=nn.Sequential(nn.Linear(60,1))
        
        self.audio_uni_layer=nn.Sequential(nn.Linear(30,1))
        
        self.image_uni_layer=nn.Sequential(nn.Linear(30,1))
        

        
    

    def forward(self, image_input:torch.Tensor, audio_input:torch.Tensor):
        # 이미지 부분 
        image_output=self.image_layer(image_input) # 콘캣 위한 fc1,2 하나
        
        # # lstm -> 오디오 부분
        # batch = audio_input.size(0)
        # audio_input = audio_input.reshape(batch, self.num_frames, self.num_feats)
        # lstm_out, _ = self.lstm(audio_input)  # (B, T, C=hidden_dim * 2)
        # feat = lstm_out.permute(0, 2, 1)  # (B, C=hidden_dim * 2, T)
        # audio_output=self.audio_layer(feat) # 콘캣 위한 fc1 하나
        
        #tssd -> 오디오 부분
        audio_input=audio_input.unsqueeze(1)
        audio_output = self.audio_tssd_layer(audio_input)
        

                
        # 이미지+ 오디오 concat 부분
        multi_input=torch.cat([image_output,audio_output],dim=1)
        multi_output=self.multi_layer(multi_input)
        
        # 이미지, 오디오 각각에 해당 Out put
        image_output=self.image_uni_layer(image_output)
        audio_output=self.audio_uni_layer(audio_output)
        return image_output, audio_output, multi_output





#random_torch=random_torch.reshape(4,2)
#random_torch.size()
# random_torch1=torch.rand(16,3,224,224)
# random_torch2=torch.rand(16,25*3,224,224)
# torch_stack=torch.cat([random_torch,random_torch1,random_torch2],dim=1) 채널을 콘캣하는 방법.?
# torch_stack.size()
##image_input=torch.rand(16,200,3,224,224)
##image_input=torch.reshape(image_input,(16,600,224,224))

# image_input=torch.rand(4,2,2,1,1)
# image_input=torch.reshape(image_input,(4,4,1,1))

##audio_input=torch.rand(16,128000)
##model=Multimodal()
##image_output, audio_output, multi_output=model(image_input,audio_input)
##print(f"image_output size {image_output.size()}, aduio_output size { audio_output.size()}, multi_output size{multi_output.size()}")
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class MultiInputModel(nn.Module):
#     def __init__(self):
#         super(MultiInputModel, self).__init__()
#         # 오디오 데이터를 처리하는 레이어
#         self.audio_layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size),
#             nn.ReLU(),
#             # 필요한 다른 레이어들 추가
#         )
#         # 이미지 데이터를 처리하는 레이어
#         self.image_layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size),
#             nn.ReLU(),
#             # 필요한 다른 레이어들 추가
#         )
#         # 오디오와 이미지 데이터를 결합하는 레이어
#         self.combine_layer = nn.Sequential(
#             nn.Linear(audio_output_size + image_output_size, combined_output_size),
#             nn.ReLU(),
#             # 필요한 다른 레이어들 추가
#         )
#         # 최종 출력 레이어
#         self.output_layer = nn.Linear(combined_output_size, num_classes)

#     def forward(self, audio_input, image_input):
#         # 오디오 데이터와 이미지 데이터를 각각 처리합니다.
#         audio_output = self.audio_layer(audio_input)
#         image_output = self.image_layer(image_input)
#         # 오디오와 이미지 데이터를 결합합니다.
#         combined_output = torch.cat((audio_output, image_output), dim=1)
#         combined_output = self.combine_layer(combined_output)
#         # 최종 출력을 생성합니다.
#         output = self.output_layer(combined_output)
#         return output

# # 모델 생성
# model = MultiInputModel()

# # Optimizer 정의
# optimizer = optim.Adam(model.parameters(), lr=0.001)