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
        # self.files=self._get_files()
        self.muilti_files=[]
        # img_files = []
        labels=[s.name for s in os.scandir(self.data_root) if s.is_dir()] #fake_real
        for label in labels : 
            if label=='real':
                file_path=os.path.join(self.data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir()] #id00018_fake
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    #TODO : 수정 : image 개수 줄이기 (몇배 줄일건지)
                    images=images[::self.select]
                    audio=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith(".wav") and not a.endswith("_d.wav") and not a.endswith("_r.wav")]
                    self.muilti_files.append((audio[0],images,label))
            #TODO: 우리데이터에 대해서 맞추기 위해서 if label=='fake' 와 label=='real'을 넣어서 구분 + 50번째줄 and d.name.endswith("_fr") 추가함
            if label=='fake':
                file_path=os.path.join(self.data_root,label)
                file_dirs=[d for d in os.scandir(file_path) if d.is_dir() and d.name.endswith("_fr")] #id00018_fake 
                for file_dir in file_dirs:
                    images=sorted([os.path.join(file_dir,i) for i in os.listdir(file_dir) if i.endswith(".jpg")],key=natural_sort_key)
                    #TODO : 수정 : image 개수 줄이기 (몇배 줄일건지)
                    images=images[::self.select]
                    audio=[os.path.join(file_dir,a) for a in os.listdir(file_dir) if a.endswith(".wav") and not a.endswith("_d.wav") and not a.endswith("_r.wav")]
                    self.muilti_files.append((audio[0],images,label))

    def __len__(self):
        return len(self.muilti_files)

    def __getitem__(self, idx):
        
        audio_file = self.muilti_files[idx][0] #path
        waveform, sample_rate = torchaudio.load(audio_file)
        img_files=self.muilti_files[idx][1]
        label = self.muilti_files[idx][2]
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
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)
        
        #TODO : 지금 이거 id_04070_fake 로 지정됨
        
        label = 1 if label=="real" else 0
        waveform = torch.mean(waveform, axis=0)

        waveform = waveform.unsqueeze(0)
        return waveform, images, label
    
    
class PadDataset(Dataset):  #TODO : vid_fps, select 변수 추가
    def __init__(self, dataset: Dataset, sec, freq, vid_fps, select, audio_type ):
        self.dataset = dataset
        self.second = sec
        self.cut = sec*freq  # max 4 sec (ASVSpoof default)
        #TODO : 보정 fps 새로정의
        self.fps=int(vid_fps/select)
        self.audio_type = audio_type
        # MFCC 변환을 초기화합니다.
        self.mfcc_transform = MFCC(
            sample_rate=16000, # 오디오 샘플의 샘플링 레이트
            n_mfcc=13, # 반환되는 MFCC의 수
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 13, "center": False}
        )
        self.lfcc_transform = LFCC(
            sample_rate=16000,
            n_lfcc=13,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False}
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        waveform = self.dataset[index][0]
        images = self.dataset[index][1]
        label = self.dataset[index][2]
        
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        images_len = images.shape[0]
        
        audio_class = self.audio_type
        
        #TODO : wave파일 repeat수 체크 필요 -> elif 써서 2개 피쳐 다르게 다뤄야할듯
        if audio_class =="lfcc":
            if (images_len >= (self.second*self.fps)) and (waveform_len>=self.cut):

                return self.lfcc_transform(waveform[: self.cut]), images[:self.second*self.fps,:], label
            
            elif (images_len >= (self.second*self.fps)) :
                
                num_wave_repeats = int(self.cut / waveform_len) + 1
                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                
                return self.lfcc_transform(padded_waveform), images[:self.second*self.fps,:], label

            elif (waveform_len>=self.cut):
                
                num_img_repeats = int((self.second*self.fps) / images_len) + 1
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]
                
                return self.lfcc_transform(waveform[: self.cut]), padded_images, label
            
            else :
                
                num_wave_repeats = int(self.cut / waveform_len) + 1
                num_img_repeats = int((self.second*self.fps) / images_len) + 1
                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]

                return self.lfcc_transform(padded_waveform), padded_images, label
            
        if audio_class =="mfcc":
            if (images_len >= (self.second*self.fps)) and (waveform_len>=self.cut):
                
                return self.mfcc_transform(waveform[: self.cut]), images[:self.second*self.fps,:], label
            
            elif (images_len >= (self.second*self.fps)) :
                
                num_wave_repeats = int(self.cut / waveform_len) + 1
                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                
                return self.mfcc_transform(padded_waveform), images[:self.second*self.fps,:], label

            elif (waveform_len>=self.cut):
                
                num_img_repeats = int((self.second*self.fps) / images_len) + 1
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]
                
                return self.mfcc_transform(waveform[: self.cut]), padded_images, label
            
            else :

                num_wave_repeats = int(self.cut / waveform_len) + 1
                num_img_repeats = int((self.second*self.fps) / images_len) + 1

                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]

                return self.mfcc_transform(padded_waveform), padded_images, label
        
        if audio_class =="wave":
            if (images_len >= (self.second*self.fps)) and (waveform_len>=self.cut):

                return waveform[: self.cut], images[:self.second*self.fps,:], label
            
            elif (images_len >= (self.second*self.fps)) :
                
                num_wave_repeats = int(self.cut / waveform_len) + 1
                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                
                return padded_waveform, images[:self.second*self.fps,:], label

            elif (waveform_len>=self.cut):
                
                num_img_repeats = int((self.second*self.fps) / images_len) + 1
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]
                
                return waveform[: self.cut], padded_images, label
            
            else :

                num_wave_repeats = int(self.cut / waveform_len) + 1
                num_img_repeats = int((self.second*self.fps) / images_len) + 1
                padded_waveform = torch.tile(waveform, (1, num_wave_repeats))[:, : self.cut][0]
                padded_images = torch.tile(images, (num_img_repeats,1,1,1))[:self.second*self.fps,:]

                return padded_waveform, padded_images, label


import numpy as np
  
data_dir="/data2/Dataset/data_us"
data=MultiDataset(root_dir=data_dir, type='test', select=5, transform=None)
padded_data=PadDataset(dataset=data, sec=6, freq=16000, vid_fps=25, select=5, audio_type='mfcc' )
multi_loader=DataLoader(padded_data, batch_size=4, shuffle=True, drop_last=True)

for audio_input , image_input , label in multi_loader:
    print(audio_input , image_input , label)
    first=audio_input[0].numpy()
    np.savez('/data1/josephlee/multimodal/asd.npz',first=first)
