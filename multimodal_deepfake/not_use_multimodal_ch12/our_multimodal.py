import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchaudio
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
from model.mul import Multimodal

import os
from torch.utils.data import Dataset,DataLoader
from torchaudio.transforms import Resample
from loader.dataload_avceleb import PadDataset,MultiDataset
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from utils import save_checkpoint, save_pred, set_learning_rate
from torch.utils.tensorboard import SummaryWriter
from model.MultiLoss import CustomLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


# directory 설정 
root="/data2/Dataset/data_us"

train_data = MultiDataset(root_dir=root, type="train",select=5, transform=None)
padded_data = PadDataset(dataset=train_data,sec=6,freq=16000,vid_fps=25,select=5)
multi_train_loader = DataLoader(padded_data, batch_size=8, shuffle=True,drop_last=True, num_workers=4)

val_data = MultiDataset(root_dir=root, type="val",select=5, transform=None)
val_padded_data = PadDataset(dataset=val_data,sec=6,freq=16000,vid_fps=25,select=5)
multi_val_loader = DataLoader(val_padded_data, batch_size=2, shuffle=True,drop_last=True, num_workers=4)

board_dir="/data1/josephlee/multimodal/tensor_log/ourdata_wave_tensorboard_one_ch/our_oneloss_ch1"

writer=SummaryWriter(log_dir=board_dir)


# setting
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

model=Multimodal().to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer,patience=10,factor=0.1,threshold=1e-5)
criterion = CustomLoss()


# 이미지 모델 초기화 및 훈련
# 멀티 모델 훈련
def train_img_audio_model(model, multi_train_loader ,optimizer, criterion, device, num_epochs=50,
                      eval_frequency=1, log_frequency=1, save_path='/data1/josephlee/multimodal/best_model/ourdata_wave_one_ch_best_model',scheduler=scheduler): # checkpoint: dict=None):
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_acc=0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_correct = 0.0
        audio_num_total = 0.0

        torch.cuda.empty_cache()
        for (audio_data, image_data, target) in multi_train_loader:
            audio_data=audio_data.to(device)
            image_data=image_data.squeeze(dim=2) 
            image_data=image_data.to(device)
            curr_batch_size=audio_data.size(0)
            audio_num_total += curr_batch_size
        
            image_output, audio_output, multi_output=model(image_data, audio_data)
            target=target.unsqueeze(1).type(torch.float32).to(device)            
            audio_loss=criterion.ce_loss(audio_output, target)
            image_loss=criterion.ce_loss(image_output, target)
            multi_loss = criterion.ce_loss(multi_output, target)

            
            loss = multi_loss + audio_loss # * 0.5  + image_loss * 0.5
            total_loss += loss
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

       # 오디오 -> original은 묶여있고     
        total_loss /= audio_num_total
        writer.add_scalar("Avg Loss update / train", total_loss, (epoch+1))
        print('train avg_loss: ',total_loss)

        # avg_loss 하나를 기준으로 봤을떄의 베스트 모델을 저장하는거로.
        if (epoch + 1) % eval_frequency == 0:
            val_avg_loss , image_avg_loss , audio_avg_loss ,  val_acc, recall, f1 , auc = evaluate_model(model, multi_val_loader, criterion) 
            writer.add_scalar("val_avg_loss update / val", val_avg_loss, (epoch+1))
            writer.add_scalar("AUC update / val", auc, (epoch+1))
            #writer.add_scalar("audio_loss update / val", audio_avg_loss, (epoch+1))  
            if (epoch + 1) % log_frequency == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], val_Avg_loss : {val_avg_loss}") # Avg Loss (Audio): {audio_avg_loss}, Avg Loss (Image): {image_avg_loss},
                print(f"Epoch [{epoch+1}/{num_epochs}], val_acc: {val_acc}, f1_score: {f1}, AUC: {auc}, recall: {recall}")
            # 최적의 모델 저장 avgLoss를 통해서 저장할것.
            if save_path and val_avg_loss < best_loss and val_acc > best_acc:
                best_loss = val_avg_loss
                best_acc = val_acc
                best_multi_model_state = model.state_dict()
                best_epoch = epoch  # 최적의 에폭을 업데이트하고
                best_model_state = {
                    'best_multi_model_state_dict': best_multi_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch+1
                }
                torch.save(best_model_state, save_path+'/our_multimodal_oneloss_1ch.pth')
                print(f"Best model saved at epoch {best_epoch+1} with val_avg_loss: {best_loss}") 
            scheduler.step(val_avg_loss)

            
        writer.flush() 
writer.close()


def evaluate_model(model, multi_val_loader ,criterion):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0
    num_correct = 0.0
    audio_num_total = 0.0
    image_num_total = 0.0
    total_audio_loss=0
    total_image_loss=0
    acu_pred=[]
    acu_target=[]
    with torch.no_grad():
        for (val_audio_data, val_image_data, val_target) in multi_val_loader:
            val_audio_data=val_audio_data.to(device)
            val_image_data=val_image_data.squeeze(dim=2) 
            val_image_data=val_image_data.to(device)
            
            acu_target.append(val_target.tolist())
            audio_curr_batch_size=val_audio_data.size(0)
            image_curr_batch_size=val_image_data.size(0) 
            image_num_total+=image_curr_batch_size
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(val_image_data,val_audio_data)
            val_target=val_target.unsqueeze(1).type(torch.float32).to(device)
            
            audio_loss=criterion.ce_loss(audio_output,val_target)
            image_loss=criterion.ce_loss(image_output,val_target)
            batch_loss=criterion.ce_loss(multi_output, val_target)

            batch_pred = (torch.sigmoid(multi_output) + 0.5).int()
            num_correct += (batch_pred ==val_target.int()).sum(dim=0).item()
            acu_pred.append(batch_pred.tolist())
            
            # accumulate loss
            total_audio_loss+=audio_loss
            #total_image_loss+=image_loss 
            total_loss += batch_loss + audio_loss  # * 0.5 + image_loss * 0.5
        
        val_acc = (num_correct / audio_num_total) * 100
        y_target = [target[0] for target in acu_target]
        y_pred = [pred[0][0] for pred in acu_pred]
        recall = recall_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred) 
        auc = roc_auc_score(y_target, y_pred)   
         
    audio_avg_loss=total_audio_loss/audio_num_total
    image_avg_loss=total_image_loss/image_num_total
    val_avg_loss = total_loss / audio_num_total
    return val_avg_loss, image_avg_loss , audio_avg_loss , val_acc, recall, f1 , auc 

    
train_img_audio_model(model,multi_train_loader, optimizer= model_optimizer, criterion=criterion, device=device)


