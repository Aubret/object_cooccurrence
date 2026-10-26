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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
            self,
            num_prototypes: int,
            warmup_teacher_temp: float,
            teacher_temp: float,
            num_epochs: int,
            warmup_teacher_temp_epochs: float = None,
            warmup_teacher_temp_steps: float = None,
            student_temp: float = 0.1,
            num_large_crops: int = 2,
            center_momentum: float = 0.9,
    ):
        """Auxiliary module to compute DINO's loss.

        Args:
            num_prototypes (int): number of prototypes.
            warmup_teacher_temp (float): base temperature for the temperature schedule
                of the teacher.
            teacher_temp (float): final temperature for the teacher.
            num_epochs (int): total number of epochs.
            num_steps (int): total number of steps.
            warmup_teacher_temp_epochs (float): number of epochs for the cosine annealing schedule.
            warmup_teacher_temp_steps (float): number of steps for the cosine annealing schedule. If
                not None, it will override `warmup_teacher_temp_epochs`. Can be the absolute number of steps or a
                fraction of the total number of steps for the whole training.
            student_temp (float, optional): temperature for the student. Defaults to 0.1.
            num_large_crops (int, optional): number of crops/views. Defaults to 2.
            center_momentum (float, optional): momentum for the EMA update of the center of
                mass of the teacher. Defaults to 0.9.
        """

        super().__init__()
        self.epoch = 0
        self.step = 0
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_large_crops = num_large_crops
        self.register_buffer("center", torch.zeros(1, num_prototypes))

        self.num_epochs = num_epochs
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.warmup_teacher_temp_steps = warmup_teacher_temp_steps

    def setup_temp_schedule(self, num_steps=None):
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        if self.warmup_teacher_temp_steps is not None:
            stepping_target = int(self.warmup_teacher_temp_steps * num_steps) if isinstance(
                self.warmup_teacher_temp_steps, float) else int(self.warmup_teacher_temp_steps)
        elif self.warmup_teacher_temp_epochs is not None:
            steps_per_epoch = num_steps // self.num_epochs
            stepping_target = self.warmup_teacher_temp_epochs * steps_per_epoch
        else:
            raise ValueError("You must provide either warmup_teacher_temp_epochs or warmup_teacher_temp_steps")

        print(f"stepping_target: {stepping_target}, stepping_max: {num_steps}")
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(self.warmup_teacher_temp, self.teacher_temp, stepping_target),
                np.ones(num_steps - stepping_target) * self.teacher_temp,
            )
        )

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """Computes DINO's loss given a batch of logits of the student and a batch of logits of the
        teacher.

        Args:
            student_output (torch.Tensor): NxP Tensor containing student logits for all views.
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits for all views.

        Returns:
            torch.Tensor: DINO loss.
        """

        # student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.num_large_crops)
        temp = self.teacher_temp_schedule[self.step]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        split_size = self.num_large_crops  # if num_large_crops is None else num_large_crops
        teacher_out = teacher_out.detach().chunk(split_size)  # if split_size > 1 else [teacher_out.detach()]

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(student_out.shape[0] // teacher_out[0].shape[0])

        # teacher centering and sharpening

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Updates the center for DINO's loss using exponential moving average.

        Args:
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits of all views.
        """

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
