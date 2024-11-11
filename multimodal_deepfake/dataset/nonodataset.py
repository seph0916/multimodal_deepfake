import torchaudio
from torchaudio.transforms import Resample
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader,Dataset
import torch

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.audio_files = self._get_audio_files()

    def _get_audio_files(self):
        audio_files = []
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.endswith(".wav"):  # 모든 오디오 파일이 .wav 형식인 것으로 가정합니다
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)

        # 필요시 resample 적용
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)
        
        # 라벨은 디렉토리 이름으로 설정
        label = os.path.basename(os.path.dirname(audio_file))
        label = 1 if label=='real' else 0
        return waveform, label
    
    
class PadDataset(Dataset):
    def __init__(self, dataset: Dataset, cut: int = 44100, label=None):
        self.dataset = dataset
        self.cut = cut  # max 4 sec (ASVSpoof default)
        self.label = label

    def __getitem__(self, index):
        waveform, label = self.dataset[index]
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        if waveform_len >= self.cut:
            if self.label is None:
                return waveform[: self.cut], label
            else:
                return waveform[: self.cut], label
        # need to pad
        num_repeats = int(self.cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, : self.cut][0]

        if self.label is None:
            return padded_waveform, label
        else:
            return padded_waveform, label

    def __len__(self):
        return len(self.dataset)