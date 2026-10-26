"""LeJEPA: Latent Embedding Joint-Embedding Predictive Architecture.

Self-supervised learning via multi-view invariance combined with a
sliced goodness-of-fit test (SIGReg) that pushes embeddings toward
an isotropic Gaussian.

References:
    Balestriero & LeCun. "LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics." 2025.
    https://arxiv.org/abs/2511.08544

Example::

    from stable_pretraining.methods import LeJEPA

    model = LeJEPA("vit_small_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 2
    all_images = [torch.randn(4, 3, 224, 224)] * 6
    model.train()
    output = model(global_images, all_images)
    output.loss.backward()

    model.eval()
    output = model(images=torch.randn(4, 3, 224, 224))
    features = output.embedding  # [N, D]
"""

from typing import Dict, Any, Sequence, List

import torch.nn as nn
from omegaconf import ListConfig
from torch.distributed.nn import all_reduce

from solo.methods import BaseMethod


import torch

from solo.utils.misc import omegaconf_select, gather_stack, gather


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: str = None,
        activation_layer=torch.nn.ReLU,
        inplace: bool = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            if in_dim is None:
                layers.append(torch.nn.LazyLinear(hidden_dim, bias=bias))
            else:
                layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer == "batch_norm":
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Projects data onto a grid of points and computes the Epps-Pulley statistic.

    :param t_max: Integration upper bound.
    :param n_points: Number of integration points.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17, world_size=1):
        super().__init__()
        assert n_points % 2 == 1

        # self._is_ddp = (
        #     torch.distributed.is_available() and torch.distributed.is_initialized()
        # )
        # assert self._is_ddp is True
        self.world_size = world_size # torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Samples [N, S] (N samples, S slices).

        :return: Per-slice statistic [S].
        """
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        # if self._is_ddp:
        #     all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
        #     all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)
        cos_mean = gather_stack(cos_mean).mean(0)
        sin_mean = gather_stack(sin_mean).mean(0)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Projects data onto random 1-D directions and averages the univariate
    Epps-Pulley statistics.  A synchronised step counter seeds the random
    projections so all DDP ranks sample identical directions.

    :param num_slices: Number of random 1-D projections.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17, world_size=1):
        super().__init__()
        # self._is_ddp = (
        #     torch.distributed.is_available() and torch.distributed.is_initialized()
        # )
        # assert self._is_ddp is True

        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points, world_size=world_size)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Embeddings [N, D].

        :return: Scalar mean EP statistic.
        """
        with torch.no_grad():
            step = self.global_step.clone()

            # if self._is_ddp:
                # All ranks increment global_step in lockstep, so this
                # broadcast is redundant under normal synchronous training.
                # It is kept as a safety net against step drift from
                # uneven batches (e.g. drop_last=False).
            torch.distributed.broadcast(step, src=0)
            # step = gather(step)

            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = x @ A
        return self.ep(proj).mean()

class LeJEPA(BaseMethod):
    '''
    Mosty copied from https://github.com/galilai-group/stable-pretraining/blob/main/stable_pretraining/methods/lejepa.py
    '''

    def __init__(
        self,
        cfg
        # encoder_name: str = "vit_base_patch16_224",
        # projector: Optional[nn.Module] = None,
        # n_slices: int = 1024,
        # t_max: float = 3.0,
        # n_points: int = 17,
        # lamb: float = 0.02,
        # pretrained: bool = False,
        # drop_path_rate: float = 0.1,
    ):
        super().__init__(cfg)

        projector = nn.Sequential(
            nn.Linear(self.features_dim, 512, bias=True),
            MLP(
                in_channels=512,
                hidden_channels=[2048, 2048, 512],
                norm_layer="batch_norm",
                activation_layer=nn.ReLU,
                inplace=True,
                dropout=0.0,
            )
        )

        cfg.method_kwargs.n_slices = omegaconf_select(cfg, "method_kwargs.n_slices", 1024)
        cfg.method_kwargs.t_max = omegaconf_select(cfg, "method_kwargs.t_max", 3)
        cfg.method_kwargs.n_points = omegaconf_select(cfg, "method_kwargs.n_points", 17)
        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.02)
        cfg.method_kwargs.modified = omegaconf_select(cfg, "method_kwargs.modified", False)
        self.world_size = len(self.cfg.devices) if isinstance(self.cfg.devices, ListConfig) else self.cfg.devices
        self.world_size *= self.cfg.num_nodes


        self.projector = projector

        self.sigreg = SlicedEppsPulley(num_slices=cfg.method_kwargs.n_slices, t_max=cfg.method_kwargs.t_max, n_points=cfg.method_kwargs.n_points, world_size=self.world_size)
        self.lamb = cfg.method_kwargs.lamb

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params


    def compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
        sigreg: SlicedEppsPulley,
        lamb: float,
        ):
        """Compute the LeJEPA loss.

        :param all_projected: All view projections [V, N, K].
        :param n_global: Number of global views.
        :param sigreg: SlicedEppsPulley module.
        :param lamb: SIGReg weight λ.
        :return: Tuple of (total_loss, inv_loss, sigreg_loss).
        """
        centers = all_projected[:n_global].mean(0)  # [N, K]
        inv_loss = (centers.unsqueeze(0) - all_projected).square().mean()

        sigreg_loss = sigreg(all_projected.reshape(-1, all_projected.size(-1)))

        loss = inv_loss + lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss


    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        out = super().multicrop_forward(X)
        # z = self.projector(out["feats"])
        # out.update({"z": z})
        return out


    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().forward(X)
        # z = self.projector(out["feats"])
        # out.update({"z": z})
        return out


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        n_augs = self.num_large_crops + self.num_small_crops

        z = self.projector(torch.cat(out["feats"]))
        bs = out["feats"][0].size(0)
        all_projected = z.view(n_augs, bs, -1)

        # bs = z.shape[1]
        # z = torch.stack(out["z"])
        # assert len(out["z"]) == n_augs
        # assert self.num_large_crops == 2

        # if self.cfg.method_kwargs.modified:
        #     print(all_projected.shape)
        loss, inv_loss, sigreg_loss = self.compute_loss(
            all_projected, self.num_large_crops, self.sigreg, self.lamb
        )
        self.log("train_loss", loss, on_step=True, sync_dist=True)
        self.log("train_inv_loss", inv_loss, on_step=True, sync_dist=True)
        self.log("train_sigreg_loss", sigreg_loss, on_step=True, sync_dist=True)
        self.log("norm", z.norm(dim=1).mean(), on_step=True, sync_dist=True)
        self.log("mean", z.mean(), on_step=True, sync_dist=True)
        self.log("std", z.std(), on_step=True, sync_dist=True)
        # if self.cfg.method_kwargs.modified:
        #     return inv_loss + class_loss
        return loss + class_loss
        # return inv_loss + class_loss