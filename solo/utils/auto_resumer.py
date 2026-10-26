from argparse import ArgumentParser
import json
import os
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from solo.utils.misc import omegaconf_select

Checkpoint = namedtuple("Checkpoint", ["creation_time", "args", "checkpoint"])


class AutoResumer:
    SHOULD_MATCH = [
        "name",
        "backbone",
        "method",
        "data.dataset",
        "max_epochs",
        "optimizer.name",
        "optimizer.batch_size",
        "optimizer.lr",
        "optimizer.weight_decay",
        "wandb.project",
        "wandb.entity",
        "pretrained_feature_extractor",
        "method_kwargs.temperature",
        "data.dataset_kwargs.gaze_size",
        "data.dataset_kwargs.time_window",
        "data.dataset_kwargs.center_crop",
        "data.dataset_kwargs.resize_gs",
    ]

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = Path("trained_models"),
        max_hours: int = 36,
    ):
        """Autoresumer object that automatically tries to find a checkpoint
        that is as old as max_time.

        Args:
            checkpoint_dir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            max_hours (int): maximum elapsed hours to consider checkpoint as valid.
        """

        self.checkpoint_dir = checkpoint_dir
        if max_hours == -1 or max_hours is None:
            self.max_hours = timedelta.max
        else:
            self.max_hours = timedelta(hours=max_hours)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.auto_resume = omegaconf_select(cfg, "auto_resume", default={})
        cfg.auto_resume.enabled = omegaconf_select(cfg, "auto_resume.enabled", default=False)
        cfg.auto_resume.max_hours = omegaconf_select(cfg, "auto_resume.max_hours", default=36)
        cfg.auto_resume.step = omegaconf_select(cfg, "auto_resume.step", default=None)
        cfg.auto_resume.epoch = omegaconf_select(cfg, "auto_resume.epoch", default=None)

        return cfg

    def find_checkpoint(self, cfg: DictConfig, step: int = None, epoch: int = None):
        """Finds a valid checkpoint that matches the arguments

        Args:
            cfg (DictConfig): DictConfig containing all settings of the model.
        """

        current_time = datetime.now()

        candidates = []
        for rootdir, _, files in os.walk(self.checkpoint_dir):
            rootdir = Path(rootdir)
            if files:
                for checkpoint_file in sorted([rootdir / f for f in files if f.endswith(".ckpt")]):
                    creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint_file))
                    if current_time - creation_time < self.max_hours:
                        ck = Checkpoint(
                            creation_time=creation_time,
                            args=rootdir / "args.json",
                            checkpoint=checkpoint_file,
                        )
                        candidates.append(ck)

        if candidates:
            # sort by most recent
            candidates = sorted(candidates, key=lambda ck: ck.creation_time, reverse=True)
            for candidate in candidates:
                if not Path(candidate.args).exists():
                    continue
                candidate_cfg = DictConfig(json.load(open(candidate.args)))
                if all(
                    omegaconf_select(candidate_cfg, param, None)
                    == omegaconf_select(cfg, param, None)
                    for param in AutoResumer.SHOULD_MATCH
                ):
                    import torch
                    ckpt = torch.load(candidate.checkpoint, map_location="cpu", weights_only=False)
                    ckpt_global_step = ckpt['global_step'] - 1
                    ckpt_epoch = ckpt['epoch']

                    if step is not None and step != ckpt_global_step:
                        continue
                    if epoch is not None and epoch != ckpt_epoch:
                        continue

                    wandb_run_id = getattr(candidate_cfg, "wandb_run_id", None)

                    print("Found", candidate.checkpoint)
                    return candidate.checkpoint, wandb_run_id

        # raise FileNotFoundError("No valid checkpoint found.")
        print("No valid checkpoint found.")
        return None, None
