import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os 
from torch.utils.data import DataLoader,Dataset
from model.channel1_mul_parse import Multimodal
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
# our 인 경우
from loader.random_loader_v2 import PadDataset,MultiDataset 
# fakeav 인 경우
#from loader.random_loader_v2_av import PadDataset,MultiDataset 
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from utils import save_checkpoint, save_pred, set_learning_rate
from torch.utils.tensorboard import SummaryWriter
from model.MultiLoss import CustomLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau




# 데이터 함수
def make_loader(data_dir, type, pad_sec, batch_size, audio_class):
    data=MultiDataset(root_dir=data_dir, type=type, select=5, transform=None)
    padded_data=PadDataset(dataset=data, sec=pad_sec, freq=16000, vid_fps=25, select=5, audio_type=audio_class )
    multi_loader=DataLoader(padded_data, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=4)
    return multi_loader

# 이미지 모델 초기화 및 훈련
# 멀티 모델 훈련
def train_img_audio_model(model, image_model, audio_model, data_dir , optimizer, criterion, device, num_epochs,
                          save_path, best_path,board_dir, scheduler, pad_sec, batch_size, audio_class, eval_frequency=1, log_frequency=1):# checkpoint: dict=None):
    best_loss=float('inf')
    best_acc=0.0
    best_epoch=0
    best_model_state=None
    multi_train_loader=make_loader(data_dir, type='train', pad_sec=pad_sec, batch_size=batch_size,audio_class=audio_class)
    writer=SummaryWriter(log_dir=board_dir)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        audio_num_total = 0.0
        acu_pred=[]
        acu_target=[]
        torch.cuda.empty_cache()
        for (audio_data, image_data, target) in multi_train_loader:
            audio_data=audio_data.to(device)
            image_data=image_data.squeeze(dim=2) # torch.reshape(image_data,(16, 40, 224, 224)) 이방법보단 저게
            image_data=image_data.to(device)
            
            curr_batch_size=audio_data.size(0)
            audio_num_total += curr_batch_size
            
            image_output, audio_output, multi_output=model(image_model, audio_model, image_data, audio_data)

            target=target.unsqueeze(1).type(torch.float32).to(device)
            multi_loss = criterion.ce_loss(multi_output, target)

            
            # accumulate loss
            loss = multi_loss 
            total_loss += loss * curr_batch_size
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        
       
       # 오디오 -> original은 묶여있고     
        total_loss /= audio_num_total
        writer.add_scalar("train Loss update / train", total_loss, (epoch+1))
        print('train avg_loss: ',total_loss)

        # avg_loss 하나를 기준으로 봤을떄의 베스트 모델을 저장하는거로.
        if (epoch + 1) % eval_frequency == 0:
            multi_val_loader=make_loader(data_dir, type='val', pad_sec=pad_sec, batch_size=batch_size, audio_class=audio_class)
            val_avg_loss , audio_avg_loss, val_acc, recall, f1, auc = evaluate_model(model, image_model, audio_model, multi_val_loader, criterion, device) #,image_avg_loss audio_avg_loss ,
            writer.add_scalar("val_avg_loss update / val", val_avg_loss, (epoch+1))
            writer.add_scalar("AUC update / val", auc, (epoch+1))  
            #writer.add_scalar("image_loss update / val", image_avg_loss, (epoch+1))
            if (epoch + 1) % log_frequency == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], val_Avg_loss : {val_avg_loss}") # Avg Loss (Image): {image_avg_loss},, Avg Loss (Audio): {audio_avg_loss}
                print(f"Epoch [{epoch+1}/{num_epochs}], val_acc: {val_acc}, f1_score: {f1}, AUC: {auc}, recall: {recall}")
            # 최적의 모델 저장 avgLoss를 통해서 저장할것.
            if save_path and val_avg_loss < best_loss and val_acc > best_acc:
                best_loss=val_avg_loss
                best_acc=val_acc
                best_multi_model_state=model.state_dict()
                best_epoch=epoch  # 최적의 에폭을 업데이트하고
                best_model_state={
                    'best_multi_model_state_dict': best_multi_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch+1,
                }
                torch.save(best_model_state, save_path + best_path)
                print(f"Best model saved at epoch {best_epoch+1} with val_avg_loss: {best_loss}")
            scheduler.step(val_avg_loss)
        writer.flush() 
    writer.close()



def evaluate_model(model, image_model, audio_model, multi_val_loader, criterion, device):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0
    num_correct = 0.0
    audio_num_total = 0.0
    image_num_total = 0.0
    total_audio_loss=0
    #total_image_loss=0
    acu_pred=[]
    acu_target=[]
    with torch.no_grad():
        for (val_audio_data, val_image_data, val_target) in multi_val_loader:
            val_audio_data=val_audio_data.to(device)
            val_image_data=val_image_data.squeeze(dim=2) # torch.reshape(val_image_data,(2, 40, 224, 224)) 이방법보단 저게
            val_image_data=val_image_data.to(device)
            
            acu_target.append(val_target.tolist())
            audio_curr_batch_size=val_audio_data.size(0)
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(image_model, audio_model, val_image_data, val_audio_data)
            val_target=val_target.unsqueeze(1).type(torch.float32).to(device)
            
            audio_loss=criterion.ce_loss(audio_output,val_target)
            multi_loss=criterion.ce_loss(multi_output, val_target)

            batch_pred=(torch.sigmoid(multi_output) + 0.5).int()
            num_correct += (batch_pred ==val_target.int()).sum(dim=0).item()
            acu_pred.append(batch_pred.tolist())
            
            # accumulate loss
            total_audio_loss+=audio_loss
            total_loss += multi_loss 
        
        val_acc = (num_correct / audio_num_total) * 100
        y_target = [target[0] for target in acu_target]
        y_pred = [pred[0][0] for pred in acu_pred]
        recall = recall_score(y_target, y_pred, average='binary', zero_division=1)
        f1 = f1_score(y_target, y_pred, average='binary', zero_division=1) 
        try:
            auc = roc_auc_score(y_target, y_pred)
        except ValueError as e:
            print(e)
            # 여기서는 에러가 발생했을 때 기본값을 사용하거나 다른 조치를 취할 수 있습니다.
            # 예를 들어, 기본값을 0.5로 설정할 수 있습니다.
            auc = 0.5   
         
    audio_avg_loss=total_audio_loss/audio_num_total
    val_avg_loss = total_loss / audio_num_total
    return val_avg_loss , audio_avg_loss, val_acc, recall, f1 , auc 

def test_model(image_model, audio_model, device, path, data_dir,  pad_sec, batch_size, audio_class):#model, image_model, audio_model, multi_test_loader, device ):
    torch.cuda.empty_cache()
    model_state=torch.load(path)
    best_multi_model_state=model_state['best_multi_model_state_dict']
    optimizer_state=model_state['optimizer_state_dict']
    device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model=Multimodal().to(device)
    model.load_state_dict(best_multi_model_state)
    model_test_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model_test_optimizer.load_state_dict(optimizer_state)
    multi_test_loader=make_loader(data_dir=data_dir, type='test', pad_sec=pad_sec, batch_size=batch_size, audio_class=audio_class)
    model.eval()
    num_correct = 0.0
    audio_num_total = 0.0
    acu_pred=[]
    acu_target=[]
    with torch.no_grad():
        for (audio_data, image_data, target) in multi_test_loader:
            audio_data=audio_data.to(device)
            image_data=image_data.squeeze(dim=2) 
            image_data=image_data.to(device)
            
            acu_target.append(target.tolist())
            audio_curr_batch_size=audio_data.size(0)
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(image_model, audio_model, image_data, audio_data)
            target=target.unsqueeze(1).type(torch.float32).to(device)

            batch_pred = (torch.sigmoid(multi_output) + 0.5).int()
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            acu_pred.append(batch_pred.tolist())

        
        test_acc = (num_correct / audio_num_total) * 100
        y_target = [target[0] for target in acu_target]
        y_pred = [pred[0][0] for pred in acu_pred]
        recall = recall_score(y_target, y_pred, average='binary', zero_division=1)
        f1 = f1_score(y_target, y_pred, average='binary', zero_division=1) 
        try:
            auc = roc_auc_score(y_target, y_pred)
        except ValueError as e:
            print(e)
            # 여기서는 에러가 발생했을 때 기본값을 사용하거나 다른 조치를 취할 수 있습니다.
            # 예를 들어, 기본값을 0.5로 설정할 수 있습니다.
            auc = 0.5
                  
    return test_acc, recall, f1 , auc
        


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        "--dir",
        help="Directory containing real data. (default: 'data//train/real')",
        type=str,
        default= "/data2/Dataset/data_us", 
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size. (default: 8)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        help="Number of maximum epochs to train. (default: 20)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--audio_class",
        help="Feature classname. (default: 'wave'), lfcc, mfcc",
        type=str,
        default="mfcc",
    )
    parser.add_argument(
        "--audio_model",
        help="Model classname. (default: 'TSSD')",
        type=str,
        default="CNN",
    )
    parser.add_argument(
        "--image_classname",
        help="Feature classname. (default: 'jpg')",
        type=str,
        default="jpg",
    )
    parser.add_argument(
        "--image_model",
        help="Model classname. (default: 'CNN')",
        type=str,
        default="CNN",
    )
    parser.add_argument(
        "--device",
        help="Device to use. (default: 'cuda' if possible)",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pad_sec",
        help="how to cut seconds (default: 6)",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--save_path",
        help="save path directory (default: " ")",
        type=str,
        default='/data1/josephlee/multimodal/best_model',
    )
    parser.add_argument(
        "--best_path",
        help="save path directory (default: " ")",
        type=str,
        default='/ourdata_mfcc_one_ch_best_model/oneloss_30',
    )    
    parser.add_argument(
        "--board_dir",
        help="save tesnsor board directory (default: " ")",
        type=str,
        default='/data1/josephlee/multimodal/tensor_log/ourdata_mfcc_tensorboard_one_ch/our_oneloss_30',
    )
    parser.add_argument(
        "--seed",
        help="Random seed. (default: 42)",
        type=int,
        default=42,
    )    

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    model=Multimodal().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=10, factor=0.1, threshold=1e-6) # default 0.0001 , 1e-5
    criterion=CustomLoss()
    train_img_audio_model(model, image_model=args.image_model, audio_model=args.audio_model, data_dir=args.data_dir, optimizer=model_optimizer, criterion=criterion,
                          device=device, num_epochs=args.epochs, save_path=args.save_path, best_path=args.best_path, board_dir=args.board_dir,
                          scheduler=scheduler, pad_sec=args.pad_sec, batch_size=args.batch_size, audio_class=args.audio_class)
    
    # path= args.save_path + args.best_path

    # test_acc, recall, f1 , auc =test_model(image_model=args.image_model, audio_model=args.audio_model, device=device,
    #                                        path=path, data_dir=args.data_dir, pad_sec=args.pad_sec, batch_size=2, audio_class=args.audio_class)
    # # 파일에 저장할 문자열 형태로 포맷팅
    # data_to_write = f"save model {path} \n Test Accuracy: {test_acc}, AUC: {auc}, Recall: {recall}, F1 Score: {f1} use model audio {args.audio_model} image {args.image_model}\n"
    # file_path = "/data1/josephlee/multimodal/result.txt"    
    # # 파일을 'append' 모드로 열어 새로운 데이터를 파일 끝에 추가
    # with open(file_path, 'a') as file:
    #     file.write(data_to_write)    

if __name__ == "__main__":
    main()