import torch
from torch import nn


import torch
import torch.nn as nn

# SEモジュールの定義
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # グローバル平均プーリング
        y = torch.mean(x, dim=(2, 3))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # 入力特徴マップにスケールを掛ける

# 改善されたCNNモデル
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

        # モデルの構造
        self.model1 = nn.Sequential(
            # 畳み込み層1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            SEBlock(32),  # SEモジュールを追加
            nn.MaxPool2d(2),

            # 畳み込み層2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),  # Dropoutで過学習を軽減
            nn.MaxPool2d(2),

            # 畳み込み層3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            SEBlock(128),  # SEモジュールを追加
            nn.MaxPool2d(2),

            # 畳み込み層4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 畳み込み層5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            SEBlock(512),  # SEモジュールを追加

            # 畳み込み層6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 畳み込み層7
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),

            # 畳み込み層8
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 畳み込み層9
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),

            # 反畳み込み層1
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層2
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層3
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層4
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層5
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層6
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),

            # 反畳み込み層7
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            # 出力層
            nn.Conv2d(64, 10, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 順伝播
        x = self.model1(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64,3,32,32))
    output = tudui(input)
    print(output.shape)


