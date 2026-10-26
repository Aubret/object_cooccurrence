import copy
import math
import os
import random
from collections import defaultdict

import torch
from lvis import LVIS
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop



class LVISCroppedObjectDataset(Dataset):
    """Dataset for cropped objects from LVIS/COCO stored in HDF5.
    Returns (cropped_image, metadata_dict) for each object.
    """
    def __init__(self, data_root, transform=None, p=0, scale=0, min_size=5000, resize=None, p_segmented=0,
                 sample_full_image=False, split="train", single=False, segm_same=True, distinct_action=False, quadruplet=False, **kwargs):
        # self.h5_file = h5py.File(os.path.join(data_root, f"lvis{'' if split == 'train' else '_val'}.h5"), 'r')
        self.path = os.path.join(data_root, f"lvis{'' if split == 'train' else '_val'}.h5")
        # self.images = self.h5_file['images']  # Variable-length byte arrays
        self.images = None

        self.triplet = distinct_action
        self.quadruplet = quadruplet
        self.p = p
        self.scale = scale
        self.sample_full_image = sample_full_image
        self.p_segmented = p_segmented
        self.min_size = min_size
        self.single=single
        self.segm_same=segm_same
        self.split = split

        # print("Start lvis", split, flush=True)
        self.lvis_api = LVIS(os.path.join(data_root, f"lvis_v1_{split}.json"))

        # Load CSV: [x, y, w, h, cat_name, img_idx, img_file, coco_url]
        self.df = pd.read_csv(
            os.path.join(data_root, f"dataset{'' if split == 'train' else '_val'}.csv"),
            header=None,
            names=['x', 'y', 'w', 'h', 'cat_name', 'img_idx', 'img_file', 'coco_url']
        )

        cat_to_id = {
            cat["name"]: cat["id"]
            for cat in self.lvis_api.load_cats(self.lvis_api.get_cat_ids())
        }

# {cat: idx for idx, cat in enumerate(sorted(self.df["cat_name"].unique()))}
        # Add the ID column
        self.df["cat_id"] = self.df["cat_name"].map(cat_to_id)

        self.df = self.df[(self.df["w"]* self.df["h"]) > min_size]
        self.df = self.df.reset_index(drop=True)

        self.pointer = defaultdict(list)
        for img_idx, group in self.df.groupby('img_idx'):
            self.pointer[img_idx] = group.index.tolist()




        self.transform = copy.deepcopy(transform)
        if resize is not None:
            self.transform.transforms[0].transform.transforms[0] = RandomResizedCrop(
                224, scale=(resize, 1), ratio=(3/4, 4/3), interpolation=transforms.InterpolationMode.BICUBIC)
            self.transform.transforms[1].transform.transforms[0] = RandomResizedCrop(
                224, scale=(resize, 1), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BICUBIC)
        print(split, len(self), flush=True)
        print(split, len(cat_to_id), flush=True)

    def __len__(self):
        return len(self.df)  # Number of cropped objects



    def get_images(self, row, img=None, p_segm=False):
        x, y, w, h = map(int, [row['x'] - self.scale*row['w']//2, row['y'] - self.scale*row['h']//2, self.scale*row['w']+row["w"], self.scale*row['h']+row["h"]])
        img_idx = int(row['img_idx'])
        if img is None:
            img = Image.open(io.BytesIO( self.images[img_idx])).convert('RGB')

        if p_segm:
            img_pair = self.get_one_segmented_image(row, img)
            if 0 not in img_pair.size:
                return img, img_pair, img_idx
        width, height = img.size

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = max(x1 + 1, min(width, x + w)), max(y1 + 1, min(height, y + h))

        img_crop = img.crop((x1, y1, x2, y2))
        # img_crop = img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
        return img, img_crop, img_idx

    def apply_mask_with_grey_bg(self, image_pil, mask_np, bg_color=128):
        image_np = np.array(image_pil)
        # grey_bg = np.full(image_np.shape, bg_color, dtype=np.uint8)
        segmented_image = image_np * mask_np[:, :, np.newaxis]#, image_np, grey_bg)
        return Image.fromarray(segmented_image)

    def get_one_segmented_image(self, row, img_full):
        img_id = row['img_file']
        ann_ids = self.lvis_api.get_ann_ids(img_ids=[img_id])
        anns = self.lvis_api.load_anns(ann_ids)
        selected_ann_ids = [a for a in anns if a["bbox"][2] * a["bbox"][3] > self.min_size]
        ann = random.choice(selected_ann_ids)
        mask = self.lvis_api.ann_to_mask(ann)
        segmented_img = self.apply_mask_with_grey_bg(img_full, mask)

        x, y, w, h = map(int, ann['bbox'])
        width, height = segmented_img.size
        cropped_segmented_img = segmented_img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
        # print(cropped_segmented_img.size, segmented_img.size, mask.shape, img_full.size, flush=True)
        return cropped_segmented_img
        # return cropped_segmented_img

    def _get_h5(self):
        # This runs inside the DataLoader worker.
        if self.images is None:
            self.images = h5py.File(self.path, "r")["images"]

    def __getitem__(self, idx):
        # Load metadata
        self._get_h5()

        row = self.df.iloc[idx]
        img_full, img, img_idx = self.get_images(row)
        if self.single:
            return self.transform(img), int(row["cat_id"])

        # Load image bytes and convert to PIL

        # Crop object

        rand = random.random()
        rand_segm = random.random()

        if rand < self.p and self.sample_full_image:
            img_pair = img_full
        # elif self.p == 0 and random.random() < self.p_segmented:
        #     img_pair = self.get_one_segmented_image(row, img_full)
        #     if 0 in img_pair.size:
        #         img_pair = img
        elif rand < self.p and len(self.pointer[img_idx]) > 1:
            new_idx = idx
            k = 0
            while new_idx == idx:
                k+=1
                new_idx = random.choice(self.pointer[img_idx])
                if k > 30:
                    print("error", new_idx, idx, self.pointer[img_idx], img_idx, flush=True)
                    break

            row = self.df.iloc[new_idx]
            _, img_pair, _ = self.get_images(row, img=img_full, p_segm=True) if rand_segm < self.p_segmented else self.get_images(row)
        else:
            _ , img_pair, _ = self.get_images(row, img=img_full, p_segm=True) if (rand_segm < self.p_segmented and self.segm_same) else (0, img, 0)

        # Metadata
        # metadata = {
        #     'category': row['cat_name'],
        #     'bbox': [x, y, w, h],
        #     'image_id': row['img_file'],
        #     'coco_url': row['coco_url'],
        #     'original_image_idx': img_idx
        # }
        if self.quadruplet:
            return self.transform(img, img, img, img_pair), int(row["cat_id"])
        if self.triplet:
            return self.transform(img, img, img_pair), int(row["cat_id"])
        return self.transform(img, img_pair), int(row["cat_id"])




class LVISPrecroped(Dataset):
    """
    Dataset for pre-cropped LVIS objects stored as JPEG bytes in HDF5.

    HDF5:
        images       : variable-length uint8 JPEG arrays
        cat_id       : LVIS category ID
        img_idx      : original image index

    The dataset supports:
        - same object pair
        - another object from same image
        - full image (if stored)
        - triplet
        - quadruplet
    """

    def __init__(
        self,
        h5_path,
        transform=None,
        p=0,
        sample_full_image=False,
        single=False,
        split="train",
        distinct_action=False,
        quadruplet=False,
        **kwargs
    ):
        # Metadata
        self.cat_ids = None
        self.img_idx = None
        self.h5_path = h5_path

        # IMPORTANT:
        # Don't open HDF5 here. Each DataLoader worker should
        # open its own handle lazily.

        h5_path = os.path.join(h5_path, f"lvis_bb_{split}_scale05_min5k.h5")
        self.h5_path = h5_path
        self.h5_file = None
        self.images = None
        self._get_h5()

        self.transform = transform

        self.p = p
        self.sample_full_image = sample_full_image
        self.single = single
        self.triplet = distinct_action
        self.quadruplet = quadruplet



        # Build pointer from metadata CSV.
        # This avoids loading the HDF5 just to construct it.
        csv_path = h5_path.replace(".h5", ".csv")
        self.df = pd.read_csv(csv_path)

        self.pointer = defaultdict(list)

        for idx, image_idx in enumerate(self.df["img_idx"]):
            self.pointer[int(image_idx)].append(idx)

        print(f"Dataset: {len(self.df)} objects")


    def __len__(self):
        return len(self.df)  # Number of cropped objects
    # ---------------------------------------------------------
    # HDF5 handling
    # ---------------------------------------------------------

    def _get_h5(self):
        """
        Open HDF5 lazily inside each DataLoader worker.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path,
                "r",
                # swmr=True,
            )

            self.images = self.h5_file["images"]
            self.cat_ids = self.h5_file["cat_id"]

            # Optional
            if "img_idx" in self.h5_file:
                self.img_idx = self.h5_file["img_idx"]
            print(len(self.cat_ids))
            print(np.max(self.cat_ids))
            print(np.min(self.cat_ids))

    # ---------------------------------------------------------
    # Image loading
    # ---------------------------------------------------------

    def _load_image(self, idx):
        # self._get_h5()

        image_bytes = self.images[idx]

        img = Image.open(
            io.BytesIO(image_bytes.tobytes())
        ).convert("RGB")

        return img

    # ---------------------------------------------------------
    # Main access
    # ---------------------------------------------------------

    def __getitem__(self, idx):

        # self._get_h5()
        img = self._load_image(idx)
        cat_id = int(self.cat_ids[idx]) -1
        if self.single:
            if self.transform is not None:
                img = self.transform(img)

            return img, cat_id

        rand = random.random()
        if rand < self.p:

            # Another object from the same original image
            img_idx = int(self.df.iloc[idx]["img_idx"])
            candidates = self.pointer[img_idx]
            if len(candidates) > 1:
                new_idx = idx
                for _ in range(30):
                    new_idx = random.choice(candidates)
                    if new_idx != idx:
                        break
                img_pair = self._load_image(new_idx)
            else:
                img_pair = img
        else:
            # Same object
            img_pair = img
        # -----------------------------------------------------
        # Transform
        # -----------------------------------------------------

        if self.quadruplet:
            # Four views
            output = self.transform(img,img,img,img_pair)
            return output, cat_id

        elif self.triplet:
            # Three views
            output = self.transform(img,img,img_pair)
            return output, cat_id
        else:
            # Two views
            output = self.transform(img,img_pair)

            return output, cat_id