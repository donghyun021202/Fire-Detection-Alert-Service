import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, WeightedRandomSampler

import librosa
import numpy as np
from pathlib import Path
from collections import Counter

from MelSpectrogram import preprocess_mel
from SirenDataset import SirenDataset
from CRNN import CRNN

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train():
    dataset = SirenDataset("./dataset")
    train_len = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    # 🔁 Sampler 제거하고, 일반 DataLoader로 대체
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    
    # 클래스 가중치 계산
    label_counts = Counter(dataset.labels)
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(len(label_counts))]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # 클래스 가중치를 반영한 손실 함수
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(25):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "siren_crnn.pt")
    print("✅ 모델 저장 완료: siren_crnn.pt")

    # 검증
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"✅ 검증 정확도: {correct / total * 100:.2f}%")
    print("FC weight example:", model.fc.weight.data[:1])

    # confusion matrix 출력
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'announcement', 'siren'],
                yticklabels=['normal', 'announcement', 'siren'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    # 테스트 시 predict만 실행
    # test_file = "test_siren.wav"
    # predict(test_file)
    train()  # 학습 시 이걸 실행
