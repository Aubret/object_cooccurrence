import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.datasets import wrap_dataset_for_transforms_v2, VOCSegmentation
    import torchvision.transforms.v2 as T
    from torchvision.tv_tensors import Mask
except ImportError:
    raise RuntimeError("Please upgrade torchvision to v0.18.0 or later. Currently installed version is ",
                       torchvision.__version__)


def prepare_transforms(dataset: str, transform_kwargs: Dict = {}) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    def replace_void_label(x):
        """ Replaces the void label 255 with 0, the background label. """
        x = x.clone()  # Ensure no in-place modifications
        x[x == 255] = 0
        return x

    print(transform_kwargs)
    img_size = transform_kwargs.get('img_size', 224)

    pascal_voc_pipeline = {
        "T_train": T.Compose([
            T.ToImage(),
            T.Lambda(replace_void_label, Mask),
            T.RandomResizedCrop(size=(img_size, img_size), antialias=True, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "T_val": T.Compose([
            T.ToImage(),
            T.Lambda(replace_void_label, Mask),
            T.Resize(size=(img_size, img_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    pipelines = {
        "PascalVOC": pascal_voc_pipeline
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
        dataset: str,
        T_train: Callable,
        T_val: Callable,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        download: bool = True,
        data_fraction: float = -1.0,
        **dataset_kwargs
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in ["PascalVOC", ]

    if dataset == "PascalVOC":
        train_dataset = VOCSegmentation(
            root=train_data_path,
            year="2012",
            image_set="train",
            download=download,
            transforms=T_train,
        )
        train_dataset = wrap_dataset_for_transforms_v2(train_dataset)

        val_dataset = VOCSegmentation(
            root=val_data_path,
            year="2012",
            image_set="val",
            download=download,
            transforms=T_val,
        )
        val_dataset = wrap_dataset_for_transforms_v2(val_dataset)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
        train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, samplers=(None, None)
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=samplers[0]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=samplers[1]
    )
    return train_loader, val_loader


def prepare_data(
        dataset: str,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        data_fraction: float = -1.0,
        auto_augment: bool = False,
        transform_kwargs: Dict = {},
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    T_train, T_val = prepare_transforms(dataset, transform_kwargs)
    if auto_augment:
        raise NotImplementedError("Auto augment not implemented yet.")

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
        **dataset_kwargs
    )

    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
