import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2


class VideoModel(nn.Module):
    def __init__(self, emb_dim=128, frozen=False):
        super().__init__()
        self.base = mobilenet_v2(pretrained=True)
        self.base.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, num_layers=1, batch_first=True)

        if frozen:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.base.features(x)
        x = self.pool(x).view(B * T, -1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return x


class AudioModel(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, emb_dim=128, num_classes=2):
        super().__init__()
        self.video_net = VideoModel(emb_dim)
        self.audio_net = AudioModel(emb_dim)
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, video, audio):
        v = self.video_net(video)
        a = self.audio_net(audio)

        if v.shape[0] != a.shape[0]:
            minb = min(v.shape[0], a.shape[0])
            v = v[:minb]
            a = a[:minb]

        x = torch.cat([v, a], dim=1)
        out = self.fusion(x)
        return out, v, a
