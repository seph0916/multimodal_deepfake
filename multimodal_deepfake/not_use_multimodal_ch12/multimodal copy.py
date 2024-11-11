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


# directory 설정 
root="/data1/data_avceleb/data_fakeav"

data = MultiDataset(root_dir=root, type="train",select=5,transform=None)
padded_data = PadDataset(dataset=data,sec=8,freq=16000,vid_fps=25,select=5)
multi_train_loader = DataLoader(padded_data, batch_size=16, shuffle=True)

val_data = MultiDataset(root_dir=root, type="val",select=5,transform=None)
val_padded_data = PadDataset(dataset=val_data,sec=8,freq=16000,vid_fps=25,select=5)
multi_val_loader = DataLoader(val_padded_data, batch_size=2, shuffle=True)

board_dir="/data1/josephlee/multimodal/tensor_log"

writer=SummaryWriter(log_dir=board_dir)

# for (test_audio_data,test_image_data, test_target) in multi_test_loader:
#     print("dd")

# setting
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=Multimodal().to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


# 이미지 모델 초기화 및 훈련
# 멀티 모델 훈련
def train_img_audio_model(model, multi_train_loader ,optimizer, criterion, device, num_epochs=50,
                      eval_frequency=1, log_frequency=1, save_path='/home/josephlee/multimodal/best_model'):#checkpoint: dict=None):
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    #start_epoch=0 , # 부분은 save + pt일때좀 확인해 봐야할듯
    
    # if checkpoint is not None:
    #     model.load_state_dict(checkpoint["state_dict"]) # 확인
    #     optim.load_state_dict(checkpoint["optimizer"])
    #     start_epoch = checkpoint["epoch"] + 1
    #     print(f"Loaded checkpoint from epoch {start_epoch - 1}")    
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_correct = 0.0
        audio_num_total = 0.0
        image_num_total = 0.0
        acu_pred=[]
        acu_target=[]
        torch.cuda.empty_cache()
        for (audio_data, image_data, target) in multi_train_loader:
            audio_data=audio_data.to(device)
            image_data=torch.reshape(image_data,(16, 40, 224, 224))
            image_data=image_data.to(device)
            
            acu_target.append(target.tolist()) #acc, f1 auc확인을 위해
            audio_curr_batch_size=audio_data.size(0)
            image_curr_batch_size=image_data.size(0)
            audio_num_total += audio_curr_batch_size
            image_num_total += image_curr_batch_size
            
            image_output, audio_output, multi_output=model(image_data, audio_data)

            target=target.unsqueeze(1).type(torch.float32).to(device)

            audio_loss=criterion(audio_output,target)
            image_loss=criterion(image_output,target)


            multi_loss = criterion(multi_output, target)
            batch_pred = (torch.sigmoid(multi_output) + 0.5).int()
            acu_pred.append(batch_pred.tolist())
            
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            # accumulate loss
            loss = multi_loss +audio_loss +image_loss
            
        
            total_loss += multi_loss + audio_loss+ image_loss
            #print("total loss :",total_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 현재 할당된 CUDA 메모리 양 확인
            # allocated_memory = torch.cuda.memory_allocated()
            # print(f"Currently allocated CUDA memory: {allocated_memory / 1024**3:.2f} GiB")

            # # 캐시된 CUDA 메모리 양 확인
            # cached_memory = torch.cuda.memory_cached()
            # print(f"Currently cached CUDA memory: {cached_memory / 1024**3:.2f} GiB")
       
       # 오디오 -> original은 묶여있고     
        total_loss /= audio_num_total
        writer.add_scalar("three Loss update / train", total_loss, (epoch+1))
        print('avg_loss: ',total_loss)

        # avg_loss 하나를 기준으로 봤을떄의 베스트 모델을 저장하는거로.
        if (epoch + 1) % eval_frequency == 0:
            avg_loss ,audio_avg_loss , image_avg_loss, val_acc, recall, f1 , auc = evaluate_model(model, multi_val_loader, criterion)
            
            if (epoch + 1) % log_frequency == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss (Audio): {audio_avg_loss}, Avg Loss (Image): {image_avg_loss}, Avg_loss : {avg_loss}")
                print(f"Epoch [{epoch+1}/{num_epochs}], val_acc: {val_acc}, f1_score: {f1}, AUC: {auc}, recall: {recall}")
            # 최적의 모델 저장 avgLoss를 통해서 저장할것.
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                best_multi_model_state = model.state_dict()
            writer.add_scalar("three Loss update / val", avg_loss, (epoch+1))
            writer.add_scalar("AUC update / val", auc, (epoch+1))    



    # 최적의 모델 저장
    if save_path:
        if best_loss > total_loss:  # 현재 손실이 이전 손실보다 작으면
            best_loss = total_loss  # 최적의 손실을 업데이트하고
            best_epoch = epoch  # 최적의 에폭을 업데이트하고
            best_model_state = {
                'best_multi_model_state_dict': best_multi_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch
            }
            torch.save(best_model_state, save_path+'/gray_40_multimodal_best_model.pth')
            print(f"Best model saved at epoch {best_epoch} with loss: {best_loss}") 
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
            val_image_data=torch.reshape(val_image_data,(2, 40, 224, 224))
            val_image_data=val_image_data.to(device)
            
            acu_target.append(val_target.tolist())
            audio_curr_batch_size=val_audio_data.size(0)
            image_curr_batch_size=val_image_data.size(0)
            image_num_total+=image_curr_batch_size
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(val_image_data,val_audio_data)
            val_target=val_target.unsqueeze(1).type(torch.float32).to(device)

            audio_loss=criterion(audio_output,val_target)
            image_loss=criterion(image_output,val_target)
            batch_loss = criterion(multi_output, val_target)
            batch_pred = (torch.sigmoid(multi_output) + 0.5).int()
            num_correct += (batch_pred ==val_target.int()).sum(dim=0).item()
            acu_pred.append(batch_pred.tolist())
            
            # accumulate loss
            total_audio_loss+=audio_loss * audio_curr_batch_size
            total_image_loss+=image_loss * image_curr_batch_size
            total_loss += batch_loss +image_loss + audio_loss
        
        val_acc = (num_correct / audio_num_total) * 100
        y_target = [target[0] for target in acu_target]
        y_pred = [pred[0][0] for pred in acu_pred]
        recall = recall_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred) 
        auc = roc_auc_score(y_target, y_pred)   
         
    audio_avg_loss=total_audio_loss/audio_num_total
    image_avg_loss=total_image_loss/image_num_total
    avg_loss = total_loss / audio_num_total
    return avg_loss ,audio_avg_loss , image_avg_loss, val_acc, recall, f1 , auc

        


# 오디오 모델 초기화 및 훈련
# 1번째 문재 optim -> Mul이라는 모델을 만듦으로써 안에 두모델을 다넣어서 쓰도록 하는거로 변경해서 옵티마이저를 하나만 사용해도됨.
# 모델 훈련 돌리기
train_img_audio_model(model,multi_train_loader, optimizer= model_optimizer, criterion=criterion, device=device)


