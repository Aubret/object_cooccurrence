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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import omegaconf_select
from solo.utils.momentum import initialize_momentum_params


class MultiLayerProj(nn.Module):
    def __init__(self, n, in_feature, hidden_feature, out_feature, *args, use_bn=True, bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(n):
            in_f = in_feature if i == 0 else hidden_feature
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_feature))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
            layers.append(nn.Linear(in_f, hidden_feature, bias=bias))
        layers.append(nn.Linear(hidden_feature if n > 0 else in_feature, out_feature))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class BYOL(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        cfg.method_kwargs.n_layer = omegaconf_select(cfg, "method_kwargs.n_layer", 2)
        cfg.method_kwargs.use_bn = omegaconf_select(cfg, "method_kwargs.use_bn", True)
        cfg.method_kwargs.n_layer_pred = omegaconf_select(cfg, "method_kwargs.n_layer_pred", 2)

        self.projector = self._build_mlp(
                cfg.method_kwargs.n_layer,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
                use_bn=cfg.method_kwargs.use_bn
            )
        #MultiLayerProj(cfg.method_kwargs.n_layer, self.features_dim, proj_hidden_dim,proj_output_dim, bias=False)
        self.momentum_projector = self._build_mlp(
                cfg.method_kwargs.n_layer,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
                use_bn=cfg.method_kwargs.use_bn
        )
        #MultiLayerProj(cfg.method_kwargs.n_layer, self.features_dim, proj_hidden_dim,proj_output_dim, bias=False)
        initialize_momentum_params(self.projector, self.momentum_projector)

        self.predictor = self._build_mlp(
                cfg.method_kwargs.n_layer_pred,
                proj_output_dim,
                proj_hidden_dim,
                proj_output_dim,
                use_bn=cfg.method_kwargs.use_bn
        )
        #MultiLayerProj(cfg.method_kwargs.n_layer_pred, proj_output_dim, pred_hidden_dim,proj_output_dim, bias=False)

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

        cfg = super(BYOL, BYOL).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

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
        p = self.predictor(z)
        out.update({"z": z, "p": p})
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
        p = self.predictor(z)
        out.update({"z": z, "p": p})
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
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
