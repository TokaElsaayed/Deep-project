import torch
import torch.nn as nn


class ResidualBlockV3(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ScratchResNet(nn.Module):
    """ResNet-style CNN trained from scratch for 200 bird classes."""

    def __init__(self, num_classes=200, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, stride=1,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList([
            self._make_layer(64, 64, blocks=2, stride=1),
            self._make_layer(64, 128, blocks=2, stride=2),
            self._make_layer(128, 256, blocks=2, stride=2),
            self._make_layer(256, 512, blocks=2, stride=2),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [
            ResidualBlockV3(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, blocks):
            layers.append(
                ResidualBlockV3(out_channels, out_channels, stride=1)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
