from typing import Type, Union

from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

from solo.utils.misc import omegaconf_select


def     init_cfg_streams(cfg):
    cfg.method_kwargs.layer_names = omegaconf_select(cfg, "method_kwargs.layer_names", ["avgpool"])

    cfg.method_kwargs.dorsal = omegaconf_select(cfg, "method_kwargs.dorsal", {})
    cfg.method_kwargs.dorsal.enabled = omegaconf_select(cfg, "method_kwargs.dorsal.enabled", False)
    cfg.method_kwargs.dorsal.in_planes = omegaconf_select(cfg, "method_kwargs.dorsal.in_planes", 2048)
    cfg.method_kwargs.dorsal.planes = omegaconf_select(cfg, "method_kwargs.dorsal.planes", 2048)
    cfg.method_kwargs.dorsal.strides = omegaconf_select(cfg, "method_kwargs.dorsal.strides", [])
    cfg.method_kwargs.dorsal.layers = omegaconf_select(cfg, "method_kwargs.dorsal.layers", 0)
    cfg.method_kwargs.dorsal.last_kernel = omegaconf_select(cfg, "method_kwargs.dorsal.last_kernel", 7)
    cfg.method_kwargs.dorsal.layers = max(cfg.method_kwargs.dorsal.layers,len(cfg.method_kwargs.dorsal.strides))
    cfg.method_kwargs.dorsal.strides = cfg.method_kwargs.dorsal.strides + [1] * (cfg.method_kwargs.dorsal.layers - len(cfg.method_kwargs.dorsal.strides))
    cfg.method_kwargs.dorsal.layer_name = cfg.method_kwargs.layer_names[0] if not cfg.method_kwargs.dorsal.enabled else cfg.method_kwargs.layer_names[-1]

    cfg.method_kwargs.ventral = omegaconf_select(cfg, "method_kwargs.ventral", {})
    cfg.method_kwargs.ventral.enabled = omegaconf_select(cfg, "method_kwargs.ventral.enabled", False)
    cfg.method_kwargs.ventral.in_planes = omegaconf_select(cfg, "method_kwargs.ventral.in_planes", 2048)
    cfg.method_kwargs.ventral.planes = omegaconf_select(cfg, "method_kwargs.ventral.planes", 2048)
    cfg.method_kwargs.ventral.strides = omegaconf_select(cfg, "method_kwargs.ventral.strides", [])
    cfg.method_kwargs.ventral.layers = omegaconf_select(cfg, "method_kwargs.ventral.layers", 0)
    cfg.method_kwargs.ventral.last_kernel = omegaconf_select(cfg, "method_kwargs.ventral.last_kernel", 7)
    cfg.method_kwargs.ventral.layers = max(cfg.method_kwargs.ventral.layers,len(cfg.method_kwargs.ventral.strides))
    cfg.method_kwargs.ventral.strides = cfg.method_kwargs.ventral.strides + [1] * (cfg.method_kwargs.ventral.layers - len(cfg.method_kwargs.ventral.strides))
    cfg.method_kwargs.ventral.layer_name = cfg.method_kwargs.layer_names[0] if not cfg.method_kwargs.ventral.enabled else cfg.method_kwargs.layer_names[1]
    return cfg

def make_layer(
        inplanes,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        strides: int,
        last_kernel
) -> nn.Sequential:
    norm_layer = nn.BatchNorm2d
    downsample = None

    layers = []

    for i in range(0, blocks):
        # if i > 0:

        if strides[i] != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, strides[i]),
                norm_layer(planes * block.expansion),
            )
        else:
            downsample = None
        layers.append(
            block(
                inplanes,
                planes,
                strides[i],
                downsample,
                base_width=64,
                norm_layer=norm_layer,
            )
        )
        inplanes = planes * block.expansion

    if last_kernel == 0:
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
    else:
        layers.append(nn.Conv2d(inplanes, planes * block.expansion, last_kernel))
        layers.append(nn.Flatten())
        layers.append(nn.BatchNorm1d(planes * block.expansion))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def create_stream(cfg_ds):
    if not cfg_ds.enabled:
        return nn.Flatten()

    dorsal = make_layer(cfg_ds.in_planes, Bottleneck, cfg_ds.planes // 4, cfg_ds.layers, strides=cfg_ds.strides,
                              last_kernel=cfg_ds.last_kernel)
    for m in dorsal.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return dorsal


