import os
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

from typing import Any, Dict, List, Sequence

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from solo.losses.simclr import simclr_loss_func, simclr_loss_func2
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select, gather


class SimCLRDouble(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        cfg.method_kwargs.use_bn = omegaconf_select(cfg, "method_kwargs.use_bn", True)
        cfg.method_kwargs.last_bn = omegaconf_select(cfg, "method_kwargs.last_bn", False)
        cfg.method_kwargs.layers = omegaconf_select(cfg, "method_kwargs.layers", 3)
        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 1)

        features = self.features_dim
        # projector
        if self.cfg.method_kwargs.alpha != 0:
            self.projector = self._build_mlp(
                    cfg.method_kwargs.layers,
                    features,
                    proj_hidden_dim,
                    proj_output_dim,
                    use_bn=cfg.method_kwargs.use_bn,
                    last_bn=cfg.method_kwargs.last_bn
                )

        self.projector_semantic = self._build_mlp(
                cfg.method_kwargs.layers,
                features,
                proj_hidden_dim,
                proj_output_dim,
                use_bn=cfg.method_kwargs.use_bn,
                last_bn=cfg.method_kwargs.last_bn
            )

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, use_bn=True, last_bn=False, first_bn=False, affine=False):
        mlp = []
        if first_bn:
            mlp.append(nn.BatchNorm1d(input_dim, affine=affine))
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(dim2))
                    mlp.append(nn.ReLU(inplace=True))
                else:
                    mlp.append(nn.GELU())
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

        cfg = super(SimCLRDouble, SimCLRDouble).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = []
        if self.cfg.method_kwargs.alpha != 0:
            extra_learnable_params.append({"name": "projector", "params": self.projector.parameters()})
        extra_learnable_params.append({"name": "projector_semantic", "params": self.projector_semantic.parameters()})
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = out["feats"]
        out.update({"o": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = out["feats"]
        out.update({"o": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        z1 = self.projector_semantic(out["o"][0])
        z2 = self.projector_semantic(out["o"][1])
        z = torch.cat((z1, z2))
        indexes = indexes.repeat(2)
        nce_loss, pos, gathered_indexes = simclr_loss_func2(z, indexes=indexes, temperature=self.temperature)

        nce_loss_semantic = 0
        if self.cfg.method_kwargs.alpha != 0:
            assert len(out["o"])  == 4, len(out["o"])
            z3 = self.projector(out["o"][1] if len(out["o"]) <= 3 else out["o"][3])
            z4 = self.projector(out["o"][2])
            z_semantic = torch.cat((z3, z4))
            nce_loss_semantic, pos_semantic , _ = simclr_loss_func2(z_semantic, indexes=indexes, temperature=self.temperature)
            self.log("train_nce_loss_semantic", nce_loss_semantic, on_epoch=True, sync_dist=True)
            self.log("positive_loss_semantic", pos_semantic.mean(), on_epoch=True, sync_dist=True)




        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        self.log("positive_loss", pos.mean(), on_epoch=True, sync_dist=True)

        return (1-self.cfg.method_kwargs.alpha)*nce_loss + class_loss + self.cfg.method_kwargs.alpha * nce_loss_semantic
