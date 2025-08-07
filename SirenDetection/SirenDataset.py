from torch.utils.data import Dataset
import torch

from pathlib import Path
import os
import librosa
import numpy as np

from collections import Counter

class SirenDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=64, duration=3):
        self.samples = []
        self.labels = []
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration

        for label, cls in enumerate(["normal", "announcement", "siren"]):
            cls_path = Path(root_dir) / cls
            for file in cls_path.glob("*.wav"):
                self.samples.append(str(file)) 
                self.labels.append(label) # 지도 학습의 일종으로 각 파일의 라벨(정답)을 함께 저장
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, _ = librosa.load(self.samples[idx], sr=self.sr)
        y = y[:self.sr * self.duration] if len(y) > self.sr * self.duration else np.pad(y, (0, self.sr * self.duration - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db + 40) / 40
        mel_tensor = torch.tensor(mel_norm).unsqueeze(0).float()
        return mel_tensor, self.labels[idx]
