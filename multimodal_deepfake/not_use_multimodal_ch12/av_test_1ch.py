import torch

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
from model.mul import Multimodal


from torch.utils.data import Dataset,DataLoader
from loader.dataload_avceleb import PadDataset,MultiDataset
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from model.MultiLoss import CustomLoss


root="/data1/data_avceleb/data_fakeav"

data = MultiDataset(root_dir=root, type="test",select=5,transform=None)
padded_data = PadDataset(dataset=data,sec=6,freq=16000,vid_fps=25,select=5)
multi_test_loader = DataLoader(padded_data, batch_size=2, shuffle=True)

path="/data1/josephlee/multimodal/best_model/avceleb_one_ch_best_model/2024_0326_avdata_onlymultiloss_1ch_multimodal.pth"
model_state=torch.load(path)
best_multi_model_state=model_state['best_multi_model_state_dict']
optimizer_state=model_state['optimizer_state_dict']
best_loss=model_state['best_loss']
best_epoch=model_state['best_epoch']
best_threshold=model_state['best_threshold']
print(best_threshold)
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=Multimodal().to(device)
model.load_state_dict(best_multi_model_state)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model_optimizer.load_state_dict(optimizer_state)
criterion = CustomLoss()


model.eval()


def evaluate_model(model, multi_test_loader ):
    torch.cuda.empty_cache()
    model.eval()
    num_correct = 0.0
    audio_num_total = 0.0
    acu_pred=[]
    acu_target=[]
    with torch.no_grad():
        for (val_audio_data, val_image_data, val_target) in multi_test_loader:
            val_audio_data=val_audio_data.to(device)
            val_image_data=val_image_data.squeeze(dim=2) 
            val_image_data=val_image_data.to(device)
            
            acu_target.append(val_target.tolist())
            audio_curr_batch_size=val_audio_data.size(0)
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(val_image_data,val_audio_data)
            val_target=val_target.unsqueeze(1).type(torch.float32).to(device)
            


            batch_pred = (torch.sigmoid(multi_output) + 0.5).int()
            num_correct += (batch_pred ==val_target.int()).sum(dim=0).item()
            acu_pred.append(batch_pred.tolist())

        
        test_acc = (num_correct / audio_num_total) * 100
        y_target = [target[0] for target in acu_target]
        y_pred = [pred[0][0] for pred in acu_pred]
        recall = recall_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred) 
        
        auc = roc_auc_score(y_target, y_pred)   



    return test_acc, recall, f1 , auc 

test_acc, recall, f1, auc =evaluate_model(model=model,multi_test_loader=multi_test_loader)
print(f"test_acc : {test_acc}, recall : {recall}, f1 : {f1}, AUC : {auc}")