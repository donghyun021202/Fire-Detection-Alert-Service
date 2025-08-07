import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # (64 → 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2))   # (32 → 16)
        )
        # CNN 출력 shape: (B, 32, 16, T/4)
        self.rnn = nn.GRU(input_size=32 * 16, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 * 2, 3)

    def forward(self, x):  # x: (B, 1, 64, T)
        x = self.cnn(x)               # -> (B, 32, 16, T/4)
        x = x.permute(0, 3, 1, 2)     # -> (B, T/4, 32, 16)
        x = x.flatten(2)              # -> (B, T/4, 512)
        out, _ = self.rnn(x)          # 🔧 input_size=512 맞춰짐
        return self.fc(out[:, -1, :])
