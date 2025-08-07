import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from CRNN import CRNN
from MelSpectrogram import preprocess_mel


def predict(file_path, model_path="siren_crnn.pt", visualize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. 오디오 로딩 및 정보 출력
    y, sr = librosa.load(file_path, sr=16000)
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    print(f"🎧 오디오 정보: RMS={rms:.5f}, Peak={peak:.5f}, Duration={len(y)/sr:.2f}s")

    # 2. 정규화 (선택)
    y = librosa.util.normalize(y)

    # 3. mel-spectrogram 시각화
    if visualize:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.title("📊 Mel-Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    # 4. 예측 수행
    mel_tensor = preprocess_mel(file_path).to(device)
    print("🔎 입력 텐서 크기:", mel_tensor.shape)

    with torch.no_grad():
        output = model(mel_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]  # [normal, announcement, siren]
        print("🔢 softmax 확률 분포:", prob)

        if prob[2] > 0.1:
            print(f"[{Path(file_path).name}] → 🚨 사이렌 감지됨 (siren)")
            return "siren"
        elif prob[1] > 0.9:
            print(f"[{Path(file_path).name}] → 📢 안내 방송 (announcement)")
            return "announcement"
        else:
            print(f"[{Path(file_path).name}] → ✅ 일반 소리 (normal)")
            return "normal"


if __name__ == '__main__':
    test_file = "test_siren.wav"
    predict(test_file)
