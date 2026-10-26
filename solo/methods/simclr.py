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


class SimCLR(BaseMethod):
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
        cfg.method_kwargs.sparse_reg = omegaconf_select(cfg, "method_kwargs.sparse_reg", 0)
        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 1)

        features = self.features_dim
        if "resnet50_dorsal" in self.cfg.backbone.name:
            features = self.features_dim // 2

        # projector
        self.projector = self._build_mlp(
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

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

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

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
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
        z = self.projector(out["feats"])
        out.update({"z": z})
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
        z = self.projector(out["feats"])
        out.update({"z": z})
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
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        # indexes = indexes.repeat(n_augs)
        indexes = indexes.repeat(2)

        nce_loss = 0
        for v1 in range(self.num_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                nce_loss_all, pos, gathered_indexes = simclr_loss_func2(
                    torch.cat((out["z"][v1], out["z"][v2])),
                    indexes=indexes,
                    temperature=self.temperature,
                )
                nce_loss += nce_loss_all
        # nce_loss += simclr_loss_func(
        #     z,
        #     indexes=indexes,
        #     temperature=self.temperature,
        # )
        # if batch_idx == 0 and self.current_epoch == 0:
        #     step, X, targets = batch
        #     img = X[0]
        #     img2 = X[1]
        #     mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        #     unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        #     os.makedirs(f"/work/autolearn/aubret/results/lvis_examples2k", exist_ok=True)
        #     print(len(img), len(img2), len(img[0]), flush=True)
        #     for i in range(32):
        #         imgi = unnormalize(img[0][i:i+1].cpu())
        #         imgi2 = unnormalize(img2[0][i:i+1].cpu())
        #         torchvision.utils.save_image(imgi, f"/work/autolearn/aubret/results/lvis_examples2k/{step[i].item()}_1.png")
        #         torchvision.utils.save_image(imgi2, f"/work/autolearn/aubret/results/lvis_examples2k/{step[i].item()}_2.png")
        #     print("finish saving")
        if self.cfg.weighted and hasattr(self, "weights"):

            indexes1 = torch.arange(len(indexes.chunk(2, 0)[0]))
            indexes_fix = indexes.chunk(2, 0)[0] - self.size_dataset0
            weights = (1-self.cfg.method_kwargs.alpha) * pos[indexes1]
            weights_all = gather(weights)
            indexes_fix_all = gather(indexes_fix)
            mask = indexes_fix_all >= 0

            indexes_fix_all = indexes_fix_all[mask]
            weights_all = weights_all[mask]
            self.weights[indexes_fix_all] = weights_all + self.cfg.method_kwargs.alpha * self.weights[indexes_fix_all]


        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        self.log("positive_loss", pos.mean(), on_epoch=True, sync_dist=True)
        self.log("sparsity", (out["feats"][0] == 0).to(torch.float).mean(), on_epoch=True, sync_dist=True)
        if self.cfg.method_kwargs.sparse_reg:
            return nce_loss + class_loss + self.cfg.method_kwargs.sparse_reg * 0.5 * (out["feats"][0].mean() + out["feats"][1].mean())


        return nce_loss + class_loss
