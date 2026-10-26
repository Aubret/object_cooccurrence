import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # In a decoder, if stride > 1, we upsample
        self.upsample = None
        if stride > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            inner_channels = out_channels
        elif in_channels != out_channels:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            inner_channels = out_channels

        else:
            inner_channels = out_channels

        # Bottleneck structure: 1x1 -> 3x3 -> 1x1
        self.conv1 = nn.Conv2d(in_channels, inner_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels // 4)

        self.conv2 = nn.Conv2d(inner_channels // 4, inner_channels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels // 4)

        self.conv3 = nn.Conv2d(inner_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.upsample is not None:
            identity = self.upsample(x)
            # Upsample the main path to match identity
            if identity.shape[2] != x.shape[2] or identity.shape[3] != x.shape[3]:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        return self.relu(out)

class ResNet50Decoder(nn.Module):
    def __init__(self, num_layers=[3, 4, 6, 3], output_channels=3, small=False):
        super().__init__()
        self.in_channels = 2048

        # Stages reverse the Encoder stages
        self.layer4 = self._make_layer(1024, num_layers[3], stride=2)
        self.layer3 = self._make_layer(512, num_layers[2], stride=2)
        self.layer2 = self._make_layer(256, num_layers[1], stride=2)
        self.layer1 = self._make_layer(64, num_layers[0], stride=1 if small else 2)
        # self.layer1 = self._make_layer(64, num_layers[0], stride=2)

        # Final upsampling to original image size
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if not small else nn.Identity(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(DecoderBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(DecoderBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.final_conv(x)
        return x