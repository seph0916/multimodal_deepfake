import torch

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
from model.channel2_mul_parse import Multimodal


from torch.utils.data import Dataset,DataLoader
from loader.random_loader_2ch import PadDataset,MultiDataset
#from loader.random_loader_2ch_av import PadDataset,MultiDataset
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from model.MultiLoss import CustomLoss





def make_loader(data_dir, type, pad_sec, batch_size, audio_class):
    data=MultiDataset(root_dir=data_dir, type=type, select=5, transform=None)
    padded_data=PadDataset(dataset=data, sec=pad_sec, freq=16000, vid_fps=25, select=5, audio_type=audio_class )
    multi_loader=DataLoader(padded_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return multi_loader








def test_model(image_model, audio_model, device, data_dir):#model, image_model, audio_model, multi_test_loader, device ):
    torch.cuda.empty_cache()
    multi_test_loader=data_dir
    model.eval()
    num_correct = 0.0
    audio_num_total = 0.0
    acu_pred=[]
    acu_target=[]
    with torch.no_grad():
        for (audio_data_d, audio_data_r, image_data, target) in multi_test_loader:
            audio_data_d=audio_data_d.to(device)
            audio_data_r=audio_data_r.to(device)
            image_data=image_data.squeeze(dim=2) 
            image_data=image_data.to(device)
            
            audio_data=torch.stack((audio_data_d, audio_data_r), dim=1)
            acu_target.append(target.tolist())
            audio_curr_batch_size=audio_data_d.size(0)
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
            auc = 0.5
                  
    return test_acc, recall, f1 , auc 



path="/data1/josephlee/multimodal/best_model/ourdata_wave_two_ch_best_model/twoloss_10"
model_state=torch.load(path)
best_multi_model_state=model_state['best_multi_model_state_dict']
optimizer_state=model_state['optimizer_state_dict']

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=Multimodal().to(device)
model.load_state_dict(best_multi_model_state)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model_optimizer.load_state_dict(optimizer_state)
# our data test root="/data2/Dataset/data_us"
# av data
root="/data2/Dataset/data_us"
multi_test_loader=make_loader(data_dir=root, type='test', pad_sec=10, batch_size=1, audio_class='wave')
model.eval()


test_acc, recall, f1, auc =test_model(image_model='CNN', audio_model="TSSD",device= device, data_dir= multi_test_loader)
data_to_write = f"save model {path} \n Test Accuracy: {test_acc}, AUC: {auc}, Recall: {recall}, F1 Score: {f1} \n"
file_path = "/data1/josephlee/multimodal/result_ch2.txt"    
# 파일을 'append' 모드로 열어 새로운 데이터를 파일 끝에 추가
with open(file_path, 'a') as file:
    file.write(data_to_write)    
