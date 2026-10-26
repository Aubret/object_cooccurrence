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

import json
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from solo.utils.misc import omegaconf_select


class Checkpointer(Callback):
    def __init__(
        self,
        cfg: DictConfig,
        logdir: Union[str, Path] = Path("trained_models"),
        frequency: int = 1,
        keep_prev: bool = False,
        save_last: bool = False,
        save_first: bool = False,
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            cfg (DictConfig): DictConfig containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            frequency (int, optional): number of epochs between each checkpoint. Defaults to 1.
            keep_prev (bool, optional): whether to keep previous checkpoints or not.
                Defaults to False.
            save_last (bool, optional): whether to save the last checkpoint or not.
        """

        super().__init__()

        self.cfg = cfg
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.keep_prev = keep_prev
        self.save_last = save_last
        self.save_first = save_first

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.checkpoint = omegaconf_select(cfg, "checkpoint", default={})
        cfg.checkpoint.enabled = omegaconf_select(cfg, "checkpoint.enabled", default=False)
        cfg.checkpoint.dir = omegaconf_select(cfg, "checkpoint.dir", default="trained_models")
        cfg.checkpoint.frequency = omegaconf_select(cfg, "checkpoint.frequency", default=1)
        cfg.checkpoint.keep_prev = omegaconf_select(cfg, "checkpoint.keep_prev", default=False)
        cfg.checkpoint.every_n_iter = omegaconf_select(cfg, "checkpoint.every_n_iter", default=0)
        cfg.checkpoint.save_last = omegaconf_select(cfg, "checkpoint.save_last", default=True)
        cfg.checkpoint.save_first = omegaconf_select(cfg, "checkpoint.save_first", default=False)
        return cfg

    @staticmethod
    def random_string(letter_count=4, digit_count=4):
        tmp_random = random.Random(time.time())
        rand_str = "".join(tmp_random.choice(string.ascii_lowercase) for _ in range(letter_count))
        rand_str += "".join(tmp_random.choice(string.digits) for _ in range(digit_count))
        rand_str = list(rand_str)
        tmp_random.shuffle(rand_str)
        return "".join(rand_str)

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if self.logdir.exists():
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = []
            version = "offline-" + self.random_string()
            while version in existing_versions:
                version = "offline-" + self.random_string()
        else:
            version = str(trainer.logger.version)
            self.wandb_run_id = version

        if version is not None:
            self.path = self.logdir / version
            self.ckpt_placeholder = f"{self.cfg.name}-{version}" + "-ep={epoch}-stp={step}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.cfg.name}" + "-ep={epoch}-stp={step}.ckpt"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer: pl.Trainer):
        """Stores arguments into a json file.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero:
            args = OmegaConf.to_container(self.cfg)
            args["wandb_run_id"] = getattr(self, "wandb_run_id", None)
            json_path = self.path / "args.json"
            json.dump(args, open(json_path, "w"), default=lambda o: "<not serializable>")

    def save(self, trainer: pl.Trainer):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """
        try:
            if not trainer.sanity_checking:
                epoch = trainer.current_epoch  # type: ignore
                ckpt = self.path / self.ckpt_placeholder.format(epoch=epoch, step=trainer.global_step)
                trainer.save_checkpoint(ckpt)

                if (
                    trainer.is_global_zero
                    and self.last_ckpt
                    and self.last_ckpt != ckpt
                    and not self.keep_prev
                ):
                    os.remove(
                        self.last_ckpt,
                    )
                self.last_ckpt = ckpt
        except Exception as e:
            print(e)
            print(ckpt)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Executes initial setup and saves arguments.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)
        self.save_args(trainer)
        if self.save_first:
            print("Saving first checkpoint")
            if not trainer.sanity_checking:
                ckpt = self.path / self.ckpt_placeholder.format(epoch='first', step='first')
                trainer.save_checkpoint(ckpt)

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Tries to save current checkpoint at the end of each train epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0:
            self.save(trainer)

    def save_iter_wise(self, trainer: pl.Trainer, batch_idx):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if not trainer.sanity_checking:
            epoch = trainer.current_epoch  # type: ignore
            ckpt = self.path / self.ckpt_placeholder.format(epoch=epoch, step=batch_idx)
            trainer.save_checkpoint(ckpt)

            if (
                trainer.is_global_zero
                and self.last_ckpt
                and self.last_ckpt != ckpt
                and not self.keep_prev
            ):
                os.remove(
                    self.last_ckpt,
                )
            self.last_ckpt = ckpt

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """
        if self.cfg.checkpoint.every_n_iter and batch_idx % self.cfg.checkpoint.every_n_iter == 0:
            self.save_iter_wise(trainer, batch_idx)



    def on_train_end(self, trainer: pl.Trainer, _):
        """Saves the last checkpoint if needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """
        if self.save_last:
            print("Saving last checkpoint")
            if not trainer.sanity_checking:
                ckpt = self.path / self.ckpt_placeholder.format(epoch='last', step='last')
                trainer.save_checkpoint(ckpt)
