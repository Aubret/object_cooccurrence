import os
import random

import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops.misc import interpolate
from torchvision.transforms import transforms, InterpolationMode


class DAVISSingleFrame(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
    ):
        """
        root: path to DAVIS
        split: train | val
        """

        root = os.path.join(root, "DAVIS-data","DAVIS")
        self.root = root
        self.transform = transform
        self.split = split
        self.target_transform = target_transform

        split_file = os.path.join(
            root, "ImageSets", "480p", f"{split}.txt"
        )

        self.samples = []

        with open(split_file, "r") as f:
            sequences = ["/".join(l.split("/")[-2:]).strip() for l in f.readlines()[0:]]

        for seq in sequences:
            img_path = os.path.join(
                root, "JPEGImages", "480p", seq
            )
            # print(img_dir,root, seq)
            # for fname in sorted(os.listdir(img_dir)):
            #     if not fname.endswith(".jpg"):
            #         continue

                # img_path = os.path.join(img_dir, fname)

            mask_path = img_path.replace("JPEGImages", "Annotations")
            img_path = img_path.replace(".png", ".jpg")
            self.samples.append((img_path, mask_path))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")





        if self.split == "train":
            crop_params = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            flip = random.random() < 0.5
            image = torchvision.transforms.functional.resized_crop(image, *crop_params, size=(384, 384))
            mask = torchvision.transforms.functional.resized_crop(mask, *crop_params, size=(384, 384), interpolation=InterpolationMode.NEAREST)

            if flip:
                image = torchvision.transforms.functional.hflip(image)
                mask = torchvision.transforms.functional.hflip(mask)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask = (mask != 0).long().unsqueeze(0)



        if len(mask.shape) > 3:
            raise Exception(mask.shape, img_path, mask_path)
        return image, mask
