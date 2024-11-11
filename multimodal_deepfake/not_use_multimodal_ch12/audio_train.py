import os
from torch.utils.data import Dataset,DataLoader
from torchaudio.transforms import Resample

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torch
import os
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from model.cnn import ShallowCNN
from model.lstm import SimpleLSTM, WaveLSTM
from model.mlp import MLP
from model.rnn import WaveRNN
from model.tssd import TSSD
from josephlee.multimodal.dataset.nonodataset import AudioDataset,PadDataset
# 일단은 보류
# KWARGS_MAP: Dict[str, dict] = {
#     "SimpleLSTM": {
#         "lfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
#         "mfcc": {"feat_dim": 40, "time_dim": 972, "mid_dim": 30, "out_dim": 1},
#     },
#     "ShallowCNN": {
#         "lfcc": {"in_features": 1, "out_dim": 1},
#         "mfcc": {"in_features": 1, "out_dim": 1},
#     },
#     "MLP": {
#         "lfcc": {"in_dim": 40 * 972, "out_dim": 1},
#         "mfcc": {"in_dim": 40 * 972, "out_dim": 1},
#     },
#     "TSSD": {
#         "wave": {"in_dim": 64600},
#     },
#     "WaveRNN": {
#         "wave": {"num_frames": 10, "input_length": 64600, "hidden_size": 500},
#     },
#     "WaveLSTM": {
#         "wave": {
#             "num_frames": 10,
#             "input_len": 64600,
#             "hidden_dim": 30,
#             "out_dim": 1,
#         }
#     },
# }

audio_data_dir='/home/josephlee/multimodal/audio/test'
audio_dataset = AudioDataset(audio_dir=audio_data_dir, transform=None)
audio_dataset = PadDataset(dataset=audio_dataset)
audio_train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
audio_model = WaveLSTM( num_frames=10,input_len= 64600,hidden_dim= 30,out_dim= 1).to(device)
optimizer = optim.Adam(audio_model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

# 오디오 모델 훈련
def train_audio_model(model, train_loader, optimizer, criterion, device, num_epochs=10,
                      eval_frequency=1, log_frequency=1, save_path='/home/josephlee/multimodal/best_model'):
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_correct = 0.0
        num_total = 0.0
        for data, target in train_loader:
            data=data.to(device)
            curr_batch_size=data.size(0)
            num_total += curr_batch_size
            output = model(data).to(device)
            target=target.unsqueeze(1).type(torch.float32).to(device)
            batch_loss = criterion(output, target)
            batch_pred = (torch.sigmoid(output) + 0.5).int()
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            # accumulate loss
            print(batch_loss)
            total_loss += batch_loss.item() * curr_batch_size
            print(total_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # 평가 및 로깅
        if (epoch + 1) % eval_frequency == 0:
            avg_loss = evaluate_model(model, train_loader, criterion)
            if (epoch + 1) % log_frequency == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss}")

            # 최적의 모델 저장
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict()
    # 최적의 모델 저장
    if save_path:
        if best_model_state:
            torch.save(best_model_state, save_path+'/audio_best_model.pth')
            print(f"Best model saved at epoch {best_epoch} with loss: {best_loss}")

def evaluate_model(model, dataloader, criterion):
    model.eval()
    num_correct = 0.0
    num_total = 0.0
    total_loss=0
    with torch.no_grad():
        for data, target in dataloader:
            data=data.to(device)
            curr_batch_size=data.size(0)
            num_total += curr_batch_size
            output = model(data).to(device)
            target=target.unsqueeze(1).type(torch.float32).to(device)
            batch_loss = criterion(output, target)
            batch_pred = (torch.sigmoid(output) + 0.5).int()
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            # accumulate loss
            print(batch_loss)
            total_loss += batch_loss.item() * curr_batch_size
            print(total_loss)
    avg_loss = total_loss / num_total
    return avg_loss

        


# 오디오 모델 초기화 및 훈련

# 모델 훈련 돌리기
train_audio_model(model= audio_model, train_loader= audio_train_loader, optimizer= optimizer, criterion= criterion, device=device)



# 이부분은 mfcc나 lfcc를 쓰려고할때 쓰면됨  
# audio_transform = transforms.Compose([
#     # 여기에 오디오 전처리를 추가합니다.
#     torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=128),
#     torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
#     torchaudio.transforms.TimeMasking(time_mask_param=100)
# ])

