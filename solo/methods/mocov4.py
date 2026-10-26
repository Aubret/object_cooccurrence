# Copyright 2023 solo-learn development team.
import os
import time
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

from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from solo.losses.mocov3 import mocov3_loss_func
from solo.methods import MoCoV3
from solo.methods.base import BaseMomentumMethod
from solo.utils.actions import get_crop_diffparams, prepare_aa_input
from solo.utils.misc import omegaconf_select, get_rank
from solo.utils.momentum import initialize_momentum_params
from solo.utils.visual_streams import create_stream, init_cfg_streams


class MoCoV4(MoCoV3):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.use_crop_params = omegaconf_select(cfg, "method_kwargs.use_crop_params", 0)
        cfg = init_cfg_streams(cfg)

        self.backbone = create_feature_extractor(self.backbone, return_nodes=list(cfg.method_kwargs.layer_names))
        self.momentum_backbone = create_feature_extractor(self.momentum_backbone, return_nodes=list(cfg.method_kwargs.layer_names))

        self.ventral = create_stream(self.cfg.method_kwargs.ventral)
        self.momentum_ventral = create_stream(self.cfg.method_kwargs.ventral)

        initialize_momentum_params(self.ventral, self.momentum_ventral)
        if get_rank() == 0:
            print(self)


    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = []
        if self.cfg.method_kwargs.ventral.enabled:
            extra_learnable_params.append({"name": "ventral", "params": self.ventral.parameters()})
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = []
        if self.cfg.method_kwargs.ventral.enabled:
            extra_momentum_pairs.append((self.ventral, self.momentum_ventral))
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = BaseMomentumMethod.forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.ventral(o)

        z = self.projector(o)
        # out.update({"q": q, "z": z})
        out.update({"z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = BaseMomentumMethod.momentum_forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.momentum_ventral(o)

        k = self.momentum_projector(o)
        out.update({"k": k})
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

        out = BaseMomentumMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]

        # MoCoV3 stuff
        Q = ( self.predictor(out["z"][0]), self.predictor(out["z"][1]))
        K = out["momentum_k"]

        contrastive_loss = mocov3_loss_func(
            Q[0], K[1], temperature=self.temperature
        ) + mocov3_loss_func(Q[1], K[0], temperature=self.temperature)

        metrics = {
            "train_contrastive_loss": contrastive_loss,
        }

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)
        loss = contrastive_loss + class_loss
        return loss

