import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample, MFCC,LFCC
from torchvision import datasets, transforms, models
import re
import PIL
import torchvision.transforms as trans
import random


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

class MultiDataset(Dataset): #TODO : select 변수 추가
    def __init__(self, root_dir, type, select , transform):
        self.data_root=os.path.join(root_dir,type) #data_fakeav/train      
        self.select= select
        self.transform=transform
        self.sample=16000
        # self.files=self._get_files()
        self.muilti_files=[]
        # img_files = []
        labels=[s.name for s in os.scandir(self.data_root) if s.is_dir()] #fake_real
        for label in labels :
            if label =='real':
                file_path=os.path.join(self.data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir()] #id00018_fake
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    audio_d=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if  a.endswith("_d.wav") ]
                    audio_r=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_r.wav") ]
                    self.muilti_files.append((audio_d[0], audio_r[0], images, label))
                    
            if label =='fake':
                file_path=os.path.join(self.data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir() and d.name.endswith("_fr")] #id00018_fake
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    #TODO : 수정 : image 개수 줄이기 (몇배 줄일건지)
                    audio_d=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if  a.endswith("_d.wav") ]
                    audio_r=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith("_r.wav") ]
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
        
        # 필요시 resample 적용
        if sample_rate_d != 16000:
            resampler = Resample(orig_freq=sample_rate_d, new_freq=16000)
            waveform_d = resampler(waveform_d)
        if sample_rate_r != 16000:
            resampler = Resample(orig_freq=sample_rate_r, new_freq=16000)
            waveform_r = resampler(waveform_r)

        
        if self.transform:
            waveform_d = self.transform(waveform_d)
            waveform_r = self.transform(waveform_r)
        
        #TODO : 지금 이거 id_04070_fake 로 지정됨
        
        label = 1 if label=="real" else 0
        waveform_d = torch.mean(waveform_d, axis=0)
        waveform_r = torch.mean(waveform_r, axis=0)
        

        # images=torch.cat(images)
        # images=images.squeeze(0)
        #TODO : 이미지 path - > data 로 변환 필요   
        
        waveform_d = waveform_d.unsqueeze(0)
        waveform_r = waveform_r.unsqueeze(0)
        return waveform_d, waveform_r, img_files, label
    
    
class PadDataset(Dataset):  #TODO : vid_fps, select 변수 추가
    def __init__(self, dataset: Dataset, sec, freq, vid_fps, select, audio_type):
        self.dataset = dataset
        self.second = sec
        self.cut = sec*freq  # max 4 sec (ASVSpoof default)
        self.select=select
        self.sample=16000
        #TODO : 보정 fps 새로정의
        self.fps=int(vid_fps/select)
        self.audio_type = audio_type
        # MFCC 변환을 초기화합니다.
        self.mfcc_transform = MFCC(
            sample_rate=16000, # 오디오 샘플의 샘플링 레이트
            # n_mfcc=13, # 반환되는 MFCC의 수
            # melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 13, "center": False}
        )
        self.lfcc_transform = LFCC(
            sample_rate=16000,
            # n_lfcc=13,
            # speckwargs={"n_fft": 400, "hop_length": 160, "center": False}
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        waveform_d = self.dataset[index][0]
        waveform_r = self.dataset[index][1]
        img_files = self.dataset[index][2]
        label = self.dataset[index][3]
        
        waveform_d = waveform_d.squeeze(0)
        waveform_r = waveform_r.squeeze(0)
        waveform_len_d = waveform_d.shape[0]
        waveform_len_r = waveform_r.shape[0]
        audio_class = self.audio_type
        waveform_d=waveform_d[ : waveform_len_r]
        images=[]
        if len(waveform_d)<=self.cut or len(img_files)<=self.fps * self.second:
            num_wave_repeats = int(self.cut / waveform_len_d) + 2
            waveform_d = torch.tile(waveform_d, (1, num_wave_repeats))[0]
            num_img_repeats = int((self.second*self.fps) / len(img_files)) + 2
            img_files = img_files * num_img_repeats
            
        if len(waveform_r)<=self.cut or len(img_files)<=self.fps * self.second:
            num_wave_repeats = int(self.cut / waveform_len_r) + 2
            
            waveform_r = torch.tile(waveform_r, (1, num_wave_repeats))[0]
            num_img_repeats = int((self.second*self.fps) / len(img_files)) + 2
            img_files = img_files * num_img_repeats
            
        max_point = len(waveform_r) - self.sample * self.second 
        start_point_wave = int(random.uniform(0,max_point))
        start_point_image = int(start_point_wave / (len(waveform_r) / len(img_files)))
        
        
        #TODO : wave파일 repeat수 체크 필요 -> elif 써서 2개 피쳐 다르게 다뤄야할듯
        waveform_d=waveform_d[start_point_wave : start_point_wave+self.cut]
        waveform_r=waveform_r[start_point_wave : start_point_wave+self.cut]
        
        # 선택할 인덱스 계산
        indices = torch.arange(start_point_image, len(img_files), self.select).tolist()
        selected_image_path = [img_files[i] for i in indices]
        selected_image_path = selected_image_path[:self.second * self.fps]

        for img_path in selected_image_path:
            img=PIL.Image.open(img_path)
            img=data_transforms(img)
            images.append(img)
        images=torch.stack(images)
                
        if images.size(0)<=self.second*self.fps:
            selected_image = torch.tile(images,(4,1,1,1))
            selected_image = selected_image[:self.second * self.fps,]
        if waveform_r.size(0)<=self.cut:
            num_wave_repeats = int(self.cut / waveform_len_r) + 1
            waveform_r = torch.tile(waveform_r, (1, num_wave_repeats))[0]          
            waveform_d = torch.tile(waveform_r, (1, num_wave_repeats))[0]
            waveform_d=waveform_d[ : self.cut]
            waveform_r=waveform_r[: self.cut]                      
        
        if audio_class =="lfcc":

                return self.lfcc_transform(waveform_d), self.lfcc_transform(waveform_r), selected_image, label
            
        if audio_class =="mfcc":
               
                return self.mfcc_transform(waveform_d), self.mfcc_transform(waveform_r), selected_image, label
        
        if audio_class =="wave":
    
                return waveform_d, waveform_r, selected_image, label


# import numpy as np
  
# data_dir="/data1/josephlee/data"
# data=MultiDataset(root_dir=data_dir, type='test', select=5, transform=None)
# padded_data=PadDataset(dataset=data, sec=30, freq=16000, vid_fps=25, select=5, audio_type='wave' )
# multi_loader=DataLoader(padded_data, batch_size=8, shuffle=True, drop_last=True)

# for audio_input_d, audio_input_r , image_input , label in multi_loader:
#     print(audio_input_d,audio_input_r , image_input , label)
#     first=audio_input[0].numpy()
#     np.savez('/data1/josephlee/multimodal/asd.npz',first=first)
