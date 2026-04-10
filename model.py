import torch.nn as nn
import torch.nn.functional as F


class ASRModel(nn.Module):
    def __init__(self, input_dim=80, num_classes=29):
        super(ASRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        rnn_input_size = 64 * 20

        self.fc = nn.Sequential(
            nn.Linear(rnn_input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
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
