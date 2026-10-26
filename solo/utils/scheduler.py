from typing import Any

import numpy as np
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import math


class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


class LogScheduler:
    def __init__(self, initial_sigma: float, final_sigma: float, decay_steps: int):
        """
        Initializes the Scheduler.

        Parameters:
        - initial_sigma: The starting sigma value.
        - final_sigma: The value sigma will decay to and stay constant at.
        - decay_steps: The number of steps over which sigma will decay.
        """
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps

        # If decaying to 0, use a small epsilon for log base calculation
        if final_sigma == 0:
            self.epsilon = 1e-8
            self.log_base = self._compute_log_base(self.epsilon)
        else:
            self.epsilon = final_sigma
            self.log_base = self._compute_log_base(final_sigma)

    def _compute_log_base(self, min_value):
        if self.decay_steps <= 1:
            return 1.0
        return math.exp((math.log(self.initial_sigma) - math.log(min_value)) / (self.decay_steps - 1))

    def __getitem__(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.final_sigma

        sigma = self.initial_sigma / (self.log_base ** step)

        # Clamp to 0 if we're decaying to 0
        if self.final_sigma == 0 and step == self.decay_steps - 1:
            return 0.0

        return max(sigma, self.final_sigma)


class GaussianBlurSigmaDecayCallback(Callback):
    def __init__(self, initial_sigma: float, final_sigma: float, decay_steps: int):
        super().__init__()
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps

        self.scheduler = None
        self.target_transforms = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # prepare augmentation
        from solo.data.pretrain_dataloader import GaussianBlur, DecayGaussianBlur

        transform = trainer.train_dataloader.dataset.transform
        self.target_transforms = transform.get_transforms(DecayGaussianBlur, transform.transforms)
        if not self.target_transforms:
            raise ValueError("GaussianBlur not found in the transform pipeline.")
        print("Found classes", self.target_transforms)

        # prepare scheduler
        _decay_steps = int(self.decay_steps * trainer.max_epochs * len(trainer.train_dataloader))
        print("_decay_steps", _decay_steps)
        self.scheduler = LogScheduler(self.initial_sigma, self.final_sigma, _decay_steps)

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ) -> None:
        value = self.scheduler[trainer.global_step]
        for T in self.target_transforms:
            T.set_sigma(value)

        pl_module.log("gaussian_blur_sigma", value, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
