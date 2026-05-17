import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=80, num_classes=29):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2)
        )
        # stride=2 applied twice means dim // 4
        freq_after_pool = max(1, input_dim // 4)
        rnn_input_size = 128 * freq_after_pool
        
        self.fc = nn.Sequential(
            nn.Linear(rnn_input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=4,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, freq, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)
        x = self.fc(x)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
