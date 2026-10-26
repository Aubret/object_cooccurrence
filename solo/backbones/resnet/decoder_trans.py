import torch
import torch.nn as nn


def conv_transpose_3x3(in_planes, out_planes, stride=1):
    """3x3 convolution transpose with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, output_padding=(stride - 1), bias=False)


class DecBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(DecBottleneck, self).__init__()
        # The order is reversed compared to the Encoder Bottleneck
        # 1. 1x1 reduction (in terms of channels, it's actually an expansion in a decoder)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 2. 3x3 upsampling convolution
        self.conv2 = conv_transpose_3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        # 3. 1x1 expansion to match the next layer's input
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50DecoderTrans(nn.Module):
    def __init__(self, latent_dim=2048, input_channels=3, small=False):
        super(ResNet50DecoderTrans, self).__init__()
        self.inplanes = 2048

        # Initial transformation from latent vector to feature map
        # For CIFAR (32x32), we start at 4x4 resolution if the encoder reduced it 3 times (32->16->8->4)
        # self.linear = nn.Linear(latent_dim, 2048 * 4 * 4)

        # Layers matching ResNet50 stages in reverse
        self.layer4 = self._make_layer(512, 3, stride=2)  # 4x4
        self.layer3 = self._make_layer(256, 6, stride=2)  # 4x4 -> 8x8
        self.layer2 = self._make_layer(128, 4, stride=2)  # 8x8 -> 16x16
        if small:
            self.layer1 = self._make_layer(64, 3, stride=1)  # 16x16 -> 32x32
            self.up_sample = nn.Identity()
        else:
            self.layer1 = self._make_layer(64, 3, stride=2)  # 16x16 -> 32x32
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final reconstruction layer
        self.conv_final = nn.Conv2d(64 * 4, input_channels, kernel_size=3, padding=1)

        # self.tanh = nn.Tanh()  # or Sigmoid depending on your normalization

    def _make_layer(self, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * DecBottleneck.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * DecBottleneck.expansion,
                                   kernel_size=1, stride=stride,
                                   output_padding=(stride - 1), bias=False),
                nn.BatchNorm2d(planes * DecBottleneck.expansion),
            )

        layers = []
        layers.append(DecBottleneck(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * DecBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(DecBottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is (Batch, Latent_Dim)
        # x = self.linear(x)
        # x = x.view(x.size(0), 2048, 4, 4)
        print(x.shape)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.up_sample(x)
        x = self.conv_final(x)
        return x