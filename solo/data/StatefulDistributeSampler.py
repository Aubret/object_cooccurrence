from typing import Union, IO, Optional, Any, Dict
from typing_extensions import Self

import numpy as np
import torch
from lightning.fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning.pytorch as pl


from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)

class DataPrepIterCheck(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.global_step = 0
    
    def setup(self, stage: str) -> None:
        pipelines = []
        for aug_cfg in self.cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(self.cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines, self.cfg.data.same_augmentations)

        if self.cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        self.train_dataset = prepare_datasets(
            self.cfg.data.dataset,
            transform,
            train_data_path=self.cfg.data.train_path,
            data_format=self.cfg.data.format,
            no_labels=self.cfg.data.no_labels,
            data_fraction=self.cfg.data.fraction,
            **self.cfg.data.dataset_kwargs
        )


    def val_dataloader(self):
        if self.cfg.data.dataset == "custom" and (self.cfg.data.no_labels or self.cfg.data.val_path is None):
            val_loader = None
        elif self.cfg.data.dataset in ["imagenet100", "imagenet", "ego4d", "nymeria","lvis_cropped"] and self.cfg.data.val_path is None:
            val_loader = None
        else:
            val_data_format = self.cfg.data.format
            _, val_loader = prepare_data_classification(
                self.cfg.data.dataset,
                train_data_path=self.cfg.data.train_path,
                val_data_path=self.cfg.data.val_path,
                data_format=val_data_format,
                batch_size=self.cfg.optimizer.batch_size,
                num_workers=self.cfg.data.num_workers,
                samplers="distributed",
                transform_kwargs=self.cfg.aug_kwargs
            )

        return val_loader

    def train_dataloader(self):
        # if self.cfg.max_epochs == 1:
        if self.cfg.weighted :
            sampler = WeightedDistributedSampler(
                self.train_dataset,
                replacement=False,
            )
        else:
            sampler = StatefulDistributedSampler(self.train_dataset, batch_size=self.cfg.optimizer.batch_size, seed=self.cfg.seed)
        train_loader = prepare_dataloader(
            self.train_dataset, batch_size=self.cfg.optimizer.batch_size, num_workers=self.cfg.data.num_workers,
            sampler=sampler, shuffle=False
        )
        # else:
        #     train_loader = prepare_dataloader(
        #         self.train_dataset, batch_size=self.cfg.optimizer.batch_size, num_workers=self.cfg.data.num_workers
        #     )
        return train_loader

    def state_dict(self) -> Dict[str, Any]:
        return {"steps": self.trainer.global_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        try:
            self.train_dataloader().sampler.set_start_iter(state_dict["steps"])
        except:
            print("cant load")

    # def load_from_checkpoint(
    #     cls,
    #     checkpoint_path: Union[_PATH, IO],
    #     map_location: _MAP_LOCATION_TYPE = None,
    #     hparams_file: Optional[_PATH] = None,
    #     **kwargs: Any,
    # ) -> Self:
    #     return super().load_from_checkpoint(cls, checkpoint_path, map_location, hparams_file, **kwargs)


class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, replacement=False, generator=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        weights = torch.ones(dataset.datasets[1].true_length)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.replacement = replacement
        self.generator = generator
        self.total_size = dataset.datasets[1].true_length - (dataset.datasets[1].true_length % self.num_replicas)
        self.second_dataset_size = 500000 #len(dataset.datasets[1])
        self.num_samples = self.second_dataset_size // self.num_replicas

        self.first_dataset_size = len(dataset.datasets[0])
        self.first_num_samples = self.first_dataset_size // self.num_replicas
        # self.first_dataset_size = len(dataset.datasets[0]) - (self.first_dataset_size % self.num_replicas)
        print(self.first_dataset_size, self.first_num_samples, self.total_size, weights.shape)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        rand_tensor = torch.multinomial(self.weights, self.second_dataset_size, self.replacement, generator=g)

        indices1 = self.first_dataset_size + rand_tensor[torch.arange(self.rank * self.num_samples, (self.rank + 1) * self.num_samples)]

        perm = torch.randperm(self.first_dataset_size, generator=g)
        indices2 = perm[self.rank * self.first_num_samples:(self.rank + 1) * self.first_num_samples]

        indices = torch.cat([indices1, indices2], dim=0)
        iterator = iter(indices[torch.randperm(len(indices), generator=g)].tolist())
        return iterator



# Taken there: https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(
                range(
                    (self.rank * self.num_samples), (self.rank + 1) * self.num_samples
                )
            )
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter

