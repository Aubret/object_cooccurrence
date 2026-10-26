# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import warnings

import omegaconf.listconfig

from solo.backbones.zoo.registry import model_registry
from solo.methods.segmentation import fix_pos_embedding

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting .* in registry with .*\. This is because the name being registered conflicts with an existing name.*",
    category=UserWarning
)
import inspect
import logging
import os

import hydra
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.linear import parse_cfg
from solo.data.classification_dataloader import prepare_data
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel, OPTICAL_FLOW_DATASETS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous, check_tensor
from torchmetrics.segmentation import MeanIoU

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

def exclude_bias_and_norm(*args, **kwargs):
    pass

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    if "pretrained" in cfg.backbone.name:
        # format fiftyone_NAME
        pretrained_backbone_name = "_".join(cfg.backbone.name.split("_")[1:])
        # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # for param in self.backbone.parameters():
        #     param.requires_grad = True
        backbone, num_features = model_registry[pretrained_backbone_name](**cfg.backbone.kwargs)
        backbone.num_features = num_features
        print("load pretrained")
    else:
        backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
        print(cfg.backbone.name)
        # initialize backbone
        backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        # cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        # if cifar or cfg.backbone.rescaled:
        #     backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        #     backbone.maxpool = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor

    # assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    if cfg.pretrained_feature_extractor is not None or cfg.resume_from_checkpoint:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        projector_state = {}

        if cfg.pretrain_method == 'dinov2':
            state = state['model']
            _filter = f"{cfg.network_type}.backbone."
        else:
            state = state["state_dict"]
            _filter = "backbone."

        for k in list(state.keys()):
            if k.startswith("projector."):
                projector_state[k.replace("projector.", "")] = state[k]
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
            if _filter in k:
                state[k.replace(_filter, "")] = state[k]
            del state[k]

        _keys = backbone.load_state_dict(state, strict=False)
        # if "vit" in cfg.backbone.name:
        #     fix_pos_embedding(backbone, (cfg.data.augmentations.img_size, cfg.data.augmentations.img_size))
        print([k for k in _keys[0] if "momentum" not in k])
        print([k for k in _keys[1] if "momentum" not in k])
        logging.info(f"Loaded {ckpt_path}")
        if cfg.use_projector:
            projector = build_mlp(len([k for k in projector_state.keys() if "weight" in k])//2 +1,  cfg.projector.proj_input_dim, cfg.projector.proj_hidden_dim, cfg.projector.proj_output_dim, True)
            projector.load_state_dict(projector_state, strict=True)
            class model(nn.Module):
                def __init__(self, backbone, head, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.backbone = backbone
                    self.head = head
                    self.num_features = 128
                def forward(self, input):
                     return self.head(self.backbone(input))
            backbone = model(backbone, projector)
    print(backbone)

    # check if mixup or cutmix is enabled
    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=cfg.data.num_classes,
        )
        # smoothing is handled with mixup label transform
        loss_func = SoftTargetCrossEntropy()
    elif cfg.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    elif cfg.data.dataset in OPTICAL_FLOW_DATASETS:#isinstance(cfg.data.num_classes, omegaconf.listconfig.ListConfig) and cfg.data.num_classes[0] == 2: #optical flow
        class OpticalLoss(nn.Module):
            def __init__(self, dataset, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dataset = dataset

            def forward(self, pred, target):
                pred = torch.nn.functional.interpolate(pred, size=target[0].shape[2:], mode='bilinear', align_corners=False)

                flow_gt = torch.nan_to_num(target[0], nan=0.0, posinf=0.0, neginf=0.0)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                diff = torch.norm(pred- flow_gt, p=2, dim=1, keepdim=True)

                if self.dataset == "FlyingThings3D":
                    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
                    valid = target[1] & (mag < 400)
                elif self.dataset == "Sintel":
                    valid = target[1]

                diff = diff * valid.float()

                epe = diff.sum() / valid.float().sum().clamp(min=1)


                check_tensor("flow_gt", target[0])
                # check_tensor("diff", masked_diff)
                check_tensor("mask", target[1])
                check_tensor("loss", epe)

                return epe
        loss_func = OpticalLoss(cfg.data.dataset)
    elif cfg.data.dataset in ["NYUv2"]: #isinstance(cfg.data.num_classes, omegaconf.listconfig.ListConfig) and cfg.data.num_classes[0] == 1:
        # assert cfg.precision==32, "Depth estimation does not work with 16 precision, probably because of interpolate (cf. torch webpage)"
        class DepthLoss(nn.Module):
            def forward(self, pred, target):
                mask = target > 0.01
                pred = torch.nn.functional.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
                return (pred[mask] - target[mask].detach()).abs().mean()
        loss_func = DepthLoss()
    elif cfg.data.dataset in ["PascalVOC","Davis"]: #isinstance(cfg.data.num_classes, omegaconf.listconfig.ListConfig) and cfg.data.num_classes[0] >= 10:
        class SegmLoss(nn.Module):
            def forward(self, pred, target):
                pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
                return torch.nn.functional.cross_entropy(pred, target.long().squeeze(), ignore_index=255)
        loss_func = SegmLoss()
    elif cfg.data.dataset == "mvimgnet2":
        class PoseLoss(nn.Module):
            def forward(self, pred, target):
                pred_rot = pred[:,:4] / (pred[:,:4].norm(dim=-1, keepdim=True)+1e-6)
                pred_trans= pred[:,4:] / (pred[:,4:].norm(dim=-1, keepdim=True)+1e-6)
                rot = (1 - torch.abs(torch.sum(pred_rot * target[:,:4], dim=-1))).mean()
                trans = torch.nn.functional.smooth_l1_loss(pred_trans, target[:,4:], reduction="mean")
                return trans + rot
        loss_func = PoseLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    model = LinearModel(backbone, loss_func=loss_func, mixup_func=mixup_func, cfg=cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
        train_backgrounds=cfg.data.train_backgrounds,
        val_backgrounds=cfg.data.val_backgrounds,
        transform_kwargs=cfg.aug_kwargs,
        **cfg.data.dataset_kwargs
    )

    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        assert not cfg.auto_augment, "Auto augmentation is not supported with Dali."

        dali_datamodule = ClassificationDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
        )

        # use normal torchvision dataloader for validation to save memory
        dali_datamodule.val_dataloader = lambda: val_loader

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id, resume_from_checkpoint = None, None, None
    if cfg.auto_resume.enabled :
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        print(cfg.auto_resume.enabled, cfg.resume_from_checkpoint, cfg.auto_resume.max_hours, resume_from_checkpoint)

        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, "linear"),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
            save_last=cfg.checkpoint.save_last,
        )
        callbacks.append(ckpt)

    if cfg.early_stopping.enabled:
        es = EarlyStopping(monitor=cfg.early_stopping.monitor, patience=cfg.early_stopping.patience,
                           mode=cfg.early_stopping.mode)
        callbacks.append(es)

    # wandb logging
    if cfg.wandb.enabled:
        d = os.environ["WANDB_DIR"] if "WANDB_DIR" in os.environ else "./"
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
            tags=cfg.wandb.tags,
            save_dir=d
        )
        # wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    callbacks.append(ModelSummary(max_depth=1))

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )

    trainer = Trainer(**trainer_kwargs)
    if cfg.validation_only:
        trainer.validate(model, dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
