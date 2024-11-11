import torch

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
from model.channel1_mul_parse import Multimodal


from torch.utils.data import Dataset,DataLoader
from loader.random_loader import PadDataset,MultiDataset
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from model.MultiLoss import CustomLoss


root="/data2/Dataset/data_us"


def make_loader(data_dir, type, pad_sec, batch_size, audio_class):
    data=MultiDataset(root_dir=data_dir, type=type, select=5, transform=None)
    padded_data=PadDataset(dataset=data, sec=pad_sec, freq=16000, vid_fps=25, select=5, audio_type=audio_class )
    multi_loader=DataLoader(padded_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return multi_loader


path="/data1/josephlee/multimodal/best_model/ourdata_wave_one_ch_best_model/oneloss"
model_state=torch.load(path)
best_multi_model_state=model_state['best_multi_model_state_dict']
optimizer_state=model_state['optimizer_state_dict']
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=Multimodal().to(device)
model.load_state_dict(best_multi_model_state)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model_optimizer.load_state_dict(optimizer_state)



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
            ssum+=ssum+val_audio_data.size(0)
            acu_target.append(val_target.tolist())
            audio_curr_batch_size=val_audio_data.size(0)
            audio_num_total+=audio_curr_batch_size
            
            image_output, audio_output, multi_output=model(image_model='CNN',audio_model='TSSD',
                                                           image_input=val_image_data, audio_input=val_audio_data)
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
multi_test_loader=make_loader(data_dir=root,type='test',pad_sec=6,batch_size=1,audio_class='wave')
test_acc, recall, f1, auc =evaluate_model(model=model,multi_test_loader=multi_test_loader)
print(f"test_acc : {test_acc}, recall : {recall}, f1 : {f1}, AUC : {auc}")