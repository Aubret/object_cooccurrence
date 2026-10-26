# Copyright 2023 solo-learn development team.
import os
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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from solo.losses.mocov3 import mocov3_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import omegaconf_select
from solo.utils.momentum import initialize_momentum_params


class MoCoV3(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements MoCo V3 (https://arxiv.org/abs/2104.02057).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim
        cfg.method_kwargs.layers = omegaconf_select(cfg, "method_kwargs.layers", 3)
        cfg.method_kwargs.layers_pred = omegaconf_select(cfg, "method_kwargs.layers_pred", 2)
        cfg.method_kwargs.tt_weight = omegaconf_select(cfg, "method_kwargs.tt_weight", 1)

        features = self.features_dim
        if "resnet50_dorsal" in self.cfg.backbone.name and hasattr(cfg.method_kwargs, "ventral"):
            features = self.features_dim // 2

        # projector
        self.projector = self._build_mlp(
            cfg.method_kwargs.layers,
            features,
            proj_hidden_dim,
            proj_output_dim,
        )
        # momentum projector
        self.momentum_projector = self._build_mlp(
            cfg.method_kwargs.layers,
            features,
            proj_hidden_dim,
            proj_output_dim,
        )

        # predictor
        self.predictor = self._build_mlp(
            cfg.method_kwargs.layers_pred,
            proj_output_dim,
            pred_hidden_dim,
            proj_output_dim,
            last_bn=False,
        )

        initialize_momentum_params(self.projector, self.momentum_projector)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, first_bn=False):
        if num_layers == 0:
            return nn.Identity()

        mlp = []
        if first_bn:
            mlp.append(nn.BatchNorm1d(input_dim))
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

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(MoCoV3, MoCoV3).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        q = self.predictor(z)
        out.update({"q": q, "z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        k = self.momentum_projector(out["feats"])
        out.update({"k": k})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MoCo V3 reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MoCo V3 and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Q = out["q"]
        K = out["momentum_k"]

        contrastive_loss = mocov3_loss_func(
            Q[0], K[1], temperature=self.temperature
        ) + mocov3_loss_func(Q[1], K[0], temperature=self.temperature)

        metrics = {
            "train_contrastive_loss": contrastive_loss,
        }
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)


        # if batch_idx == 0:
        #     step, X, targets = batch
        #     img = X[0]
        #     img2 = X[1]
        #     mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        #     unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        #     os.makedirs(f"/scratch_shared/aubret/results/kitti", exist_ok=True)
        #     print(len(img), len(img2), len(img[0]), flush=True)
        #     for i in range(32):
        #         imgi = unnormalize(img[0][i:i+1].cpu())
        #         imgi2 = unnormalize(img2[0][i:i+1].cpu())
        #         torchvision.utils.save_image(imgi, f"/scratch_shared/aubret/results/kitti/{step[i].item()}_1.png")
        #         torchvision.utils.save_image(imgi2, f"/scratch_shared/aubret/results/kitti/{step[i].item()}_2.png")
        #     print("finish saving")

        return  self.cfg.method_kwargs.tt_weight * contrastive_loss + class_loss

