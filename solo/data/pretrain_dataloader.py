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

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Type, Union

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import Compose

from solo.data.cortical_magnification_downsample import CorticalMagnificationDown
from solo.data.custom.base import H5ClassificationDataset
from solo.data.custom.imagenet import ImgnetDataset, Ecoset, ImgNetDataset_42, ImgnetContextDataset, \
    ImgNetDatasetContext_42
from solo.data.custom.mvimgnet import MVImgNet
from solo.data.custom.objects365 import Objects365, Objects365Crops
from solo.data.custom.places import Places
from solo.data.custom.tinyimgnet import TinyDataset

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class DecayGaussianBlur:
    def __init__(self, sigma:float):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """
        from multiprocessing import Value
        import ctypes

        self.sigma = Value(ctypes.c_float, sigma)

    def set_sigma(self, value):
        """Sets the sigma value for the gaussian blur.

        Args:
            value (float): sigma value.
        """
        self.sigma.value = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'GaussianBlur(sigma={self.sigma})'

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """
        sigma = self.sigma.value
        if sigma <= 0:
            return img

        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'GaussianBlur(sigma={self.sigma})'

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        if not isinstance(self.sigma, (float, int)):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma

        if sigma <= 0:
            return img

        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class GreyGaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        if not isinstance(self.sigma, (float, int)):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)

class ToTensorSafe:
    def __init__(self):
        self.t = transforms.ToTensor()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return self.t(x)

class HorizontalFlipWithParams:
    dim = 1
    def __call__(self, img):
        r = random.random() < 0
        if r:
            img = transforms.functional.hflip(img)
        return img, torch.tensor([r])

class RandomResizedCropWithParams:
    dim = 4
    def __init__(self, size, scale, ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=transforms.InterpolationMode.BICUBIC):
        self.scale = scale
        self.ratio = ratio
        self.size = size if isinstance(size, tuple) else (size, size)
        self.crop = transforms.RandomResizedCrop(size=size, scale=scale, ratio=self.ratio, interpolation=interpolation)

    def __call__(self, img):
        w, h = img.size
        params = self.crop.get_params(img, scale=self.scale, ratio=self.ratio)
        tparams = torch.tensor(params, dtype=torch.float32)
        out = transforms.functional.crop(img, *params)
        out = transforms.functional.resize(out, self.size, interpolation=transforms.functional.InterpolationMode.BICUBIC)

        # print(tparams)
        tparams[0] = (tparams[0] + 0.5*tparams[2])/ h
        tparams[1] = (tparams[1] + 0.5*tparams[3])/ w
        tparams[2] = tparams[2] / h
        tparams[3] = tparams[3] / w
        # print(tparams)
        return out, tparams

class ComposedAugsWithParams(transforms.Compose):

    def __call__(self, img):
        params = []
        for t in self.transforms:
            img = t(img)
            if isinstance(img, tuple):
                img, params_temp = img
                params.append(params_temp)
        if params:
            return img, torch.cat(params)
        return [img, params]

class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class GreenMagentaProjection(object):
    """
    Implements the 'projection' preprocessing from
    Unsupervised Visual Representation Learning by Context Prediction.
    Removes the green–magenta axis from RGB.
    """

    def __init__(self):
        # a = [-1, 2, -1]
        a = torch.tensor([-1.0, 2.0, -1.0]).view(3, 1)

        # Compute B = I - a a^T / (a^T a)
        I = torch.eye(3)
        denom = (a.t() @ a).item()  # scalar
        self.B = (I - (a @ a.t()) / denom).float()

    def __call__(self, img):
        # PIL → tensor (C,H,W)
        # t = torch.transforms.functional.to_tensor(img)

        # Reshape to (3, H*W), multiply by B, then reshape back

        C, H, W = img.shape
        t2 = self.B @ img.view(3, -1)
        t2 = t2.view(C, H, W)

        t2 = torch.clamp(t2, 0, 1)

        return t2
        # return torch.transforms.functional.to_pil_image(t2)

class ColorDrop(torch.nn.Module):
    """
    Color dropping from:
    Doersch et al., 2015 (Context Prediction)

    - Randomly keep 1 of the 3 RGB channels
    - Replace the other 2 with Gaussian noise
    - Noise std = 1/100 of the kept channel std
    """

    def __init__(self, p=1.0, noise_scale=0.01):
        """
        p: probability of applying the transform
        noise_scale: fraction of kept-channel std for noise (default 0.01)
        """
        super().__init__()
        self.p = p
        self.noise_scale = noise_scale

    def forward(self, img):
        """
        img: Tensor (3, H, W), float
        """
        if torch.rand(1) > self.p:
            return img

        if img.ndim != 3 or img.size(0) != 3:
            raise ValueError("ColorDrop expects image of shape (3, H, W)")

        # choose channel to keep
        keep = torch.randint(0, 3, (1,)).item()
        drop = [c for c in range(3) if c != keep]

        # compute noise std
        std = img[keep].std()
        if std == 0:
            std = 1e-6

        noise = torch.randn_like(img[keep]) * (self.noise_scale * std)

        out = img.clone()
        for c in drop:
            out[c] = noise

        return out

class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image, x2: Image = None, x3=None, action2=None, action=None, **kwargs) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        if x2 is not None:
            return [self.transform(random.choice([x, x2])) for _ in range(self.num_crops)]
        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable, same_augmentations=False) -> None:
        self.transforms = transforms
        self.same_augmentations = same_augmentations
        if self.same_augmentations == 2:
            self.crop_resize = torchvision.transforms.RandomResizedCrop(
                224,
                scale=(0.2, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )
        if self.same_augmentations == 3:
            self.crop_resize = RandomResizedCropWithParams(
                96,
                scale=(0.2, 0.2),
                ratio=(1, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )


    @staticmethod
    def get_transforms(cls: type, transforms: List[Callable]) -> List[Callable]:
        """Recursively checks a list of transforms and returns a specific class.

        Args:
            cls (Type): class of the transformation.
            transforms (List[Callable]): list of transformations.

        Returns:
            Optional[Callable]: transformation of the specific class.
        """
        matched = []
        if isinstance(transforms, Compose):
            transforms = transforms.transforms

        for t in transforms:
            if isinstance(t, cls):
                matched.append(t)

            if hasattr(t, 'transform'):
                matched.extend(FullTransformPipeline.get_transforms(cls, t.transform))
            if hasattr(t, 'transforms'):
                matched.extend(FullTransformPipeline.get_transforms(cls, t.transforms))

        return matched

    def __call__(self, x: Image, x2: Image = None, x3: Image = None, x4: Image=None, action2 = None, action=None) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        out = []
        if x2 is not None and x3 is not None and x4 is not None:
            out.extend(self.transforms[0](x))
            out.extend(self.transforms[1](x2))
            out.extend(self.transforms[2](x3))
            out.extend(self.transforms[2](x4))

        elif x2 is not None and x3 is not None:
            if self.same_augmentations == 2:
                x = self.crop_resize(x)
                x2 = self.crop_resize(x2)
                x3 = self.crop_resize(x3)

            state = torch.get_rng_state()
            out.extend(self.transforms[0](x))
            if self.same_augmentations:
                torch.set_rng_state(state)
            out.extend(self.transforms[1](x2))
            if self.same_augmentations:
                torch.set_rng_state(state)
            out.extend(self.transforms[1](x3))
            for t in self.transforms[3:]:
                # out.extend(transform(random.choice([x, x2])))
                out.extend(t(random.choice([x, x2, x3])))
        elif x2 is not None:
            if self.same_augmentations in [2,3]:
                x = self.crop_resize(x)
                x2 = self.crop_resize(x2)

            state = torch.get_rng_state()
            out.extend(self.transforms[0](x))
            if self.same_augmentations:
                torch.set_rng_state(state)
            out.extend(self.transforms[1](x2))
            for t in self.transforms[2:]:
                # out.extend(transform(random.choice([x, x2])))
                out.extend(t(random.choice([x, x2])))
        else:
            if self.same_augmentations in [2, 3]:
                xall = [self.crop_resize(x) for _ in self.transforms]
                state = torch.get_rng_state()
                for transform, xi in zip(self.transforms, xall):
                    torch.set_rng_state(state)
                    out.extend([[transform(xi[0])[0][0], xi[1]]])
            else:
                for transform in self.transforms:
                    out.extend(transform(x))

        if action2 is not None:
            out.append(action2)
        if action is not None:
            out.append(action)
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)


def build_transform_pipeline(dataset, cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet2_100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet2": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "kitti": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "ecoset": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "mvimgnet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet_context": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "Places365_h5": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    crop_func = transforms.RandomResizedCrop
    if hasattr(cfg.rrc, "params") and cfg.rrc.params:
        crop_func = RandomResizedCropWithParams


    augmentations = []

    if hasattr(cfg, "global_gaussian_blur") and cfg.global_gaussian_blur.enabled:
        print("ADDING global_gaussian_blur wiht sigma", cfg.global_gaussian_blur.sigma)
        augmentations.append(DecayGaussianBlur(sigma=cfg.global_gaussian_blur.sigma))

    if hasattr(cfg, "greyscaled_global_gaussian_blur") and cfg.greyscaled_global_gaussian_blur.enabled:
        print("ADDING greyscaled_global_gaussian_blur wiht sigma", cfg.greyscaled_global_gaussian_blur.sigma)
        augmentations.append(GreyGaussianBlur(sigma=cfg.greyscaled_global_gaussian_blur.sigma))

    ratio = (3/4, 4/3)
    if hasattr(cfg, "cmd") and cfg.cmd.enabled:
        ratio = (1,1)
    if cfg.rrc.enabled:
        augmentations.append(
            crop_func(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                ratio=ratio if cfg.rrc.crop_min_scale != cfg.rrc.crop_max_scale else (1, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        augmentations.append(transforms.CenterCrop(cfg.crop_size))

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.gaussian_blur.prob:
        augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

    if cfg.solarization.prob:
        augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

    if cfg.horizontal_flip.prob:
        if hasattr(cfg.horizontal_flip, "params") and cfg.horizontal_flip.params:
            augmentations.append(HorizontalFlipWithParams())
        else:
            augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

    augmentations.append(ToTensorSafe())
    if hasattr(cfg, "green_magenta_proj") and cfg.green_magenta_proj.enabled:
        augmentations.append(GreenMagentaProjection())

    if hasattr(cfg, "color_drop") and cfg.color_drop:
        augmentations.append(ColorDrop())

    if hasattr(cfg, "cmd") and cfg.cmd.enabled:
        augmentations.append(CorticalMagnificationDown(cfg.crop_size,cfg.crop_size,cfg.cmd.fovea_size, cfg.cmd.output_size))

    augmentations.append(transforms.Normalize(mean=mean, std=std))

    # augmentations = transforms.Compose(augmentations)
    augmentations = ComposedAugsWithParams(augmentations)
    return augmentations


def prepare_n_crop_transform(
        transforms: List[Callable], num_crops_per_aug: List[int],
        same_augmentations=False
):
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T, same_augmentations=same_augmentations)


def prepare_datasets(
        dataset: str,
        transform: Callable,
        train_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        no_labels: Optional[Union[str, Path]] = False,
        download: bool = True,
        data_fraction: float = -1.0,
        train_data_path_second = None,
        **dataset_kwargs
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )
    elif dataset in ["lvis_cropped"]:
        # from solo.data.custom.lvis import LVISCroppedObjectDataset
        # train_dataset = dataset_with_index(LVISCroppedObjectDataset)(train_data_path, transform, **dataset_kwargs)
        from solo.data.custom.lvis import LVISPrecroped
        train_dataset = dataset_with_index(LVISPrecroped)(train_data_path, transform, **dataset_kwargs)
    elif dataset in ["lvis_cropped_i1k"]:
        from solo.data.custom.lvis import LVISCroppedObjectDataset
        train_dataset = dataset_with_index(torch.utils.data.ConcatDataset)(
            [LVISCroppedObjectDataset(train_data_path, transform, **dataset_kwargs),
             ImgnetDataset(str(train_data_path_second), "train", transform, False)]
        )

        # train_dataset = torch.utils.data.ConcatDataset([LVISCroppedObjectDataset(train_data_path, transform, **dataset_kwargs),
        #                                                                 ImgnetDataset(str(train_data_path_second), "train", transform,False)])
    elif dataset in ["i1k_places"]:
        train_dataset = dataset_with_index(torch.utils.data.ConcatDataset)(
            [ImgnetDataset(str(train_data_path_second), "train", transform, False),
            Places(root=Path(train_data_path) / 'Places365', transform=transform, split="train", **dataset_kwargs)]
        )
    elif dataset in ["o365"]:
        train_dataset = dataset_with_index(Objects365Crops)(str(train_data_path), transform=transform, **dataset_kwargs)
    elif dataset in ["i1k_o365"]:
        path = Path(train_data_path_second) / 'ImageNet'
        if path.exists():
            subset = dataset_kwargs.get("subset")
            train_dataset = dataset_with_index(torch.utils.data.ConcatDataset)(
                [ImgNetDataset_42(Path(train_data_path_second) / 'ImageNet/h5', transform,split="train", subset=subset),
                Objects365Crops(str(train_data_path), transform=transform,  **dataset_kwargs)])
        else:
            train_dataset = dataset_with_index(torch.utils.data.ConcatDataset)(
                [ImgnetDataset(str(train_data_path_second), "train", transform, False),
                # Objects365(str(train_data_path), transform=transform,  **dataset_kwargs)]
                Objects365Crops(str(train_data_path), transform=transform,  **dataset_kwargs)]
            )
    elif dataset in ["kitti"]:
        from solo.data.custom.kitti import Kitti
        train_dataset = dataset_with_index(Kitti)(train_data_path, transform, **dataset_kwargs)
    elif dataset == "tiny":
        train_dataset = dataset_with_index(TinyDataset)(train_data_path, split="train", transform=transform)
    elif dataset == "Places365_h5":
        train_dataset = dataset_with_index(H5ClassificationDataset)(root=Path(train_data_path) / 'Places365', transform=transform,split="train")
    elif dataset in ["imagenet_42", "imagenet100_42", "imagenet1pct_42", "imagenet10pct_42", "imagenet100_im","imagenet_im"]:
        if dataset == "imagenet100_42":
            subset = "imgnet100"
        elif dataset == "imagenet100_im":
            subset = 'imgnet100_im'
        elif dataset == "imagenet1pct_42":
            subset = "1pct"
        elif dataset == "imagenet10pct_42":
            subset = "10pct"
        else:
            subset = None

        train_dataset = dataset_with_index(ImgNetDataset_42)(Path(train_data_path) / 'ImageNet/h5', transform, split="train", subset=subset)
    elif dataset in ["imagenet_im_context"]:
        train_dataset = dataset_with_index(ImgNetDatasetContext_42)(Path(train_data_path) / 'ImageNet/h5', transform, split="train", **dataset_kwargs)

    elif dataset in ["imagenet2", "imagenet2_100"]:
        train_dataset = dataset_with_index(ImgnetDataset)(train_data_path, "train", transform,
                                                          dataset == "imagenet2_100")
    elif dataset in ["imagenet_context"]:
        train_dataset = dataset_with_index(ImgnetContextDataset)(train_data_path, "train", transform, **dataset_kwargs)
    elif dataset in ["ecoset", "ecoset"]:
        train_dataset = dataset_with_index(Ecoset)(Path(train_data_path) / 'ecoset'/'h5', transform, split="train")
    elif dataset == "mvimgnet":
        train_dataset = dataset_with_index(MVImgNet)(Path(train_data_path), transform, split="train", subset=0)
    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        else:
            train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)

    elif dataset == "custom":
        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.images
            (
                files,
                _,
            ) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.images = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
        train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, sampler=None, shuffle=True
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )
    return train_loader
