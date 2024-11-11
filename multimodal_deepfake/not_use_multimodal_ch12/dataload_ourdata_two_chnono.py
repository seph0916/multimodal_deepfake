import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from torchvision import datasets, transforms, models
import re

import PIL
import torchvision.transforms as trans



def natural_sort_key(s):
    """주어진 문자열에 대한 자연 정렬 키를 생성하는 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# image_datasets = datasets.ImageFolder(image_test_data_dir, transform=data_transforms)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1)
])

#path init - > get item audio, image 

class MultiDataset(Dataset): #TODO : set_num 변수추가 --> 우리 훈련데이터의 set number표시

    def __init__(self, root_dir, type, select , set_num , transform):
        self.select= select
        self.transform=transform
        self.muilti_files=[]
        self.data_root=os.path.join(root_dir,type) #data_fakeav/train      
        
        #TODO : set_num 변수는 train일 때만 사용
        if type == "train":
            set_data_root = os.path.join(self.data_root,f"set{set_num}")
            labels=[s.name for s in os.scandir(set_data_root) if s.is_dir()] #fake_real
            for label in labels : 
                file_path=os.path.join(set_data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir()] #id00018_fake
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    #TODO : 수정 : image 개수 줄이기 (몇배 줄일건지)
                    images=images[::self.select]


                    audio_d=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_d.wav")]
                    audio_r=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_r.wav")]
                    self.muilti_files.append((audio_d[0], audio_r[0], images, label))            
        else :
        # img_files = []
            labels=[s.name for s in os.scandir(self.data_root) if s.is_dir()] #fake_real
            for label in labels : 
                file_path=os.path.join(self.data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir()] #id00018_fake
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    #TODO : 수정 : image 개수 줄이기 (몇배 줄일건지)
                    images=images[::self.select]


                    audio_d=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_d.wav")]
                    audio_r=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_r.wav")]
                    self.muilti_files.append((audio_d[0], audio_r[0], images, label))
        
    def __len__(self):
        return len(self.muilti_files)

    def __getitem__(self, idx):
        
        audio_file_d = self.muilti_files[idx][0] #path
        audio_file_r = self.muilti_files[idx][1]
        waveform_d, sample_rate_d = torchaudio.load(audio_file_d)
        waveform_r, sample_rate_r = torchaudio.load(audio_file_r)
        img_files=self.muilti_files[idx][2]
        label = self.muilti_files[idx][3]
        images=[]
        for img_path in img_files:
            img=PIL.Image.open(img_path)
            img=data_transforms(img)
            images.append(img)
        images=torch.stack(images)
        # images=torch.cat(images)
        # images=images.squeeze(0)
        #TODO : 이미지 path - > data 로 변환 필요
        
        # 필요시 resample 적용
        if sample_rate_d != 16000 :
            resampler = Resample(orig_freq=sample_rate_d, new_freq=16000)
            waveform_d = resampler(waveform_d)

        if self.transform:
            waveform_d = self.transform(waveform_d)
        
        if sample_rate_r != 16000 :
            resampler = Resample(orig_freq=sample_rate_r, new_freq=16000)
            waveform_r = resampler(waveform_r)

        if self.transform:
            waveform_r = self.transform(waveform_r)   
        
        #TODO : 지금 이거 id_04070_fake 로 지정됨
        
        label = 1 if label=="real" else 0
        waveform_d = torch.mean(waveform_d, axis=0)
        waveform_r = torch.mean(waveform_r, axis=0)

        waveform_d = waveform_d.unsqueeze(0)
        waveform_r = waveform_r.unsqueeze(0)
        return waveform_d, waveform_r, images, label
    
    
class PadDataset(Dataset):  #TODO : vid_fps, select 변수 추가
    def __init__(self, dataset: Dataset, sec, freq, vid_fps, select ):
        self.dataset = dataset
        self.second=sec
        self.cut = int(sec*freq)  # max 4 sec (ASVSpoof default)

        #TODO : 보정 fps 새로정의
        self.fps=int(vid_fps/select)

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):

        waveform_d = self.dataset[index][0]
        waveform_r = self.dataset[index][1]
        images = self.dataset[index][2]
        label = self.dataset[index][3]
        
        waveform_d = waveform_d.squeeze(0)
        waveform_r = waveform_r.squeeze(0)
        waveform_len_d = waveform_d.shape[0]
        waveform_len_r = waveform_r.shape[0]
        images_len=images.shape[0]
        #TODO : wave파일 repeat수 체크 필요 -> elif 써서 2개 피쳐 다르게 다뤄야할듯
        if (images_len >= (self.second*self.fps)) and (waveform_len_d>=self.cut) and (waveform_len_r>=self.cut):

            return waveform_d[: self.cut], waveform_r[: self.cut], images[:self.second*self.fps,:], label
        
        elif (images_len >= (self.second*self.fps)) :
            
            num_wave_repeats = int(self.cut / waveform_len_d) + 1
            padded_waveform_d = torch.tile(waveform_d, (1, num_wave_repeats))[:, : self.cut][0]
            num_wave_repeats = int(self.cut / waveform_len_r) + 1
            padded_waveform_r = torch.tile(waveform_r, (1, num_wave_repeats))[:, : self.cut][0]
            return padded_waveform_d, padded_waveform_r, images[:self.second*self.fps,:], label

        elif (waveform_len_d>=self.cut) and (waveform_len_r>=self.cut):
            num_img_repeats = int((self.second*self.fps) / images_len) + 1
            padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]
            return waveform_d[: self.cut], waveform_r[:self.cut], padded_images, label
        
        else :

            num_wave_repeats = int(self.cut / waveform_len_d) + 1
            padded_waveform_d = torch.tile(waveform_d, (1, num_wave_repeats))[:, : self.cut][0]
            
            num_wave_repeats = int(self.cut / waveform_len_r) + 1
            padded_waveform_r = torch.tile(waveform_r, (1, num_wave_repeats))[:, : self.cut][0]
            
            num_img_repeats = int((self.second*self.fps) / images_len) + 1
            padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]


            return padded_waveform_d, padded_waveform_r, padded_images, label



# root="/home/minkyo/Dataset/us"

# train_data = MultiDataset(root_dir=root, type="train",select=5, set_num=5,transform=None)
# padded_data = PadDataset(dataset=train_data,sec=6,freq=16000,vid_fps=25,select=5)
# multi_train_loader = DataLoader(padded_data, batch_size=8, shuffle=True,drop_last=True)


# for (audio_data_d, audio_data_r, image_data, target) in multi_train_loader:
#     print(audio_data_d.size(), audio_data_r.size(), image_data.size(), target.size())