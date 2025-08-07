import librosa
import numpy as np
import torch

def preprocess_mel(file_path, sr=16000, n_mels=64, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 40) / 40
    mel_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 64, T)
    return mel_tensor