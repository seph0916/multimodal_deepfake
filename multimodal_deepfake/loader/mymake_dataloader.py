import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

 
transform=transforms.Compose(transforms.Resize((224,224)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             transforms.Grayscale())
 
data_dir='/data1/josephlee/multimodal/data/train'
type='train'
#
for root, dir, path in os.walk(data_dir):
    print(path)


# 디렉토리 내에 디렉토리들 이름을 가져오는 부분.    

for a in os.listdir(data_dir):
    os.path.isdir(data_dir)
    


# class ImageDataset(Dataset):
#     def __init__(self, data_dir, transfrom, type):
#         self.root_dir=os.path.join(data_dir, type)
#         self.transform=transform
#         for i in os.listdir(self.root_dir):
#             print(i)
#     def __len__(self):
#         return len()
    
#     def __getitem__(self, index):