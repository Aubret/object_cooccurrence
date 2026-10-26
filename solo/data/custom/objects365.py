import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import io
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

class Objects365(Dataset):
    def __init__(self, data_root, transform=None, p=0, scale=0, min_size=10000, min_scale=0.2, max_scale=1,
                 p_segmented=0, sample_full_image=False, max_size=0, scaled=True, **kwargs):
        self.images = h5py.File(os.path.join(data_root, "train.h5" if not scaled else "train_640.h5"), 'r')["images"]
        self.p = p
        self.scale = scale
        self.sample_full_image = sample_full_image
        self.p_segmented = p_segmented
        self.min_size = min_size
        self.max_size= max_size


        # Load CSV: [x, y, w, h, cat_name, img_idx, img_file, coco_url]
        self.mapping = json.load(open(os.path.join(data_root, "mapping.json")))
        self.dataset = json.load(open(os.path.join(data_root, "zhiyuan_objv2_train.json")))

        image_dict = {r["id"] :r  for r in self.dataset["images"]}
        rows = []
        errors = 0
        print("ks", [k for i, k in enumerate(self.mapping.keys()) if i < 10])
        for ann in self.dataset["annotations"]:
            if str(ann["image_id"]) in self.mapping:
                scale = 1
                if scaled:
                    scale = min(1.0, 640 / max(image_dict[ann["image_id"]]["width"], image_dict[ann["image_id"]]["height"]))

                rows.append({
                    "img_idx": self.mapping[str(ann["image_id"])],
                    "image_id": str(ann["image_id"]),
                    # "category_id": ann["category_id"],
                    "x": int(scale * ann["bbox"][0]),
                    "y": int(scale * ann["bbox"][1]),
                    "w": int(scale * ann["bbox"][2]),
                    "h": int(scale * ann["bbox"][3]),
                })
            else:
                if errors < 10:
                    print(ann["image_id"])
                errors += 1
        print("nerr", errors)

        self.df = pd.DataFrame(rows)

        self.df = self.df[(self.df["w"]* self.df["h"]) > min_size]
        self.df = self.df.reset_index(drop=True)

        self.pointer = defaultdict(list)
        for img_idx, group in self.df.groupby('img_idx'):
            self.pointer[img_idx] = group.index.tolist()

        self.transform = copy.deepcopy(transform)
        self.transform.transforms[0].transform.transforms[0] = RandomResizedCrop(
            224, scale=(min_scale, max_scale), ratio=(3/4, 4/3), interpolation=transforms.InterpolationMode.BICUBIC)
        self.transform.transforms[1].transform.transforms[0] = RandomResizedCrop(
            224, scale=(min_scale, max_scale), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BICUBIC)
        self.true_length = len(self.df)

        print(self.true_length)
        print("Length:", len(self))

    def __len__(self):
        return self.true_length if not self.max_size else self.max_size


    def get_images(self, row):
        x, y, w, h = map(int, [row['x'] - self.scale*row['w']//2, row['y'] - self.scale*row['h']//2, self.scale*row['w']+row["w"], self.scale*row['h']+row["h"]])


        img_idx = int(row['img_idx'])
        # t = time.time()
        img_bytes = self.images[img_idx]
        # t2 = time.time()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # t3 = time.time()

        width, height = img.size
        # assert , f"{x}, {width}, {y}, {height}, {w}, {h}"
        # assert max(0, y) < min(height, y + h), f"{x}, {width}, {y}, {height}, {w}, {h}"

        if not (max(0, x) < min(width, x + w)) or not (max(0, y) < min(height, y + h)):
            return img, img, img_idx
        img_crop = img.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h)))
        # t4 = time.time()
        # print("in get images", t2-t, t3-t2, t4-t3, width, height, flush=True)
        return img, img_crop, img_idx

    def apply_mask_with_grey_bg(self, image_pil, mask_np, bg_color=128):
        image_np = np.array(image_pil)
        # grey_bg = np.full(image_np.shape, bg_color, dtype=np.uint8)
        segmented_image = image_np * mask_np[:, :, np.newaxis]#, image_np, grey_bg)
        return Image.fromarray(segmented_image)


    def __getitem__(self, idx):
        # print("start iteration", idx, flush=True)
        if self.max_size:
            idx = (idx + np.random.randint(0, self.true_length // self.max_size) * self.max_size)
        # Load metadata
        # print(idx, flush=
        # t = time.time()
        row = self.df.iloc[idx]
        # t2 = time.time()
        img_full, img, img_idx = self.get_images(row)
        # t3=time.time()
        # Load image bytes and convert to PIL

        # Crop object
        img_pair = img

        rand = random.random()

        if rand < self.p and self.sample_full_image:
            img_pair = img_full
        # elif rand < self.p and random.random() < self.p_segmented:
        #     img_pair = self.get_one_segmented_image(row, img_full)
        #     if 0 in img_pair.size:
        #         img_pair = img
        elif len(self.pointer[img_idx]) > 1 and rand < self.p:
            new_idx = idx
            while new_idx == idx:
                new_idx = random.choice(self.pointer[img_idx])

            row = self.df.iloc[new_idx]
            _, img_pair, _ = self.get_images(row)
        # t4 = time.time()
        # print("out get images", t2-t, t3-t2, t4-t3, flush=True)
        # Metadata
        # metadata = {
        #     'category': row['cat_name'],
        #     'bbox': [x, y, w, h],
        #     'image_id': row['img_file'],
        #     'coco_url': row['coco_url'],
        #     'original_image_idx': img_idx
        # }
        # print("end iteration", idx, flush=True)
        return self.transform(img, img_pair), idx



class Objects365Crops(Dataset):
    def __init__(self, data_root, transform=None, p=1.0, max_size=0, min_scale=0.2, max_scale=1, pairs=False,
                 sort_model="clip", threshold=0.1, **kwargs):

        self.h5 = h5py.File(os.path.join(data_root, "objects365_crops.h5"), 'r')
        self.mapping = json.load(open(os.path.join(data_root, "mapping.json")))

        self.images = self.h5["images"]
        self.image_ids = self.h5["image_ids"]
        self.max_size = max_size
        self.p=p
        self.pairs = pairs

        # Build image_id -> crop indices
        print("Before pointer", flush=True)

        self.pointer = defaultdict(list)
        for idx, image_id in enumerate(self.image_ids):
            self.pointer[int(image_id)].append(idx)

        l_mean = 0
        for p in self.pointer.values():
            l_mean += len(p)
        l_mean /= len(self.pointer)

        self.true_length = len(self.images)
        self.transform = copy.deepcopy(transform)
        self.transform.transforms[0].transform.transforms[0] = RandomResizedCrop(
            224, scale=(min_scale, max_scale), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BICUBIC)
        self.transform.transforms[1].transform.transforms[0] = RandomResizedCrop(
            224, scale=(min_scale, max_scale), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BICUBIC)

        if self.pairs:
            self.dataset = pd.read_csv(os.path.join(data_root, "pairs.csv"),header=None)
            if sort_model != "clip_inv":
                self.dataset = self.dataset[self.dataset.loc[:,3] < threshold] if sort_model == "clip" else self.dataset.loc[self.dataset.loc[:,4] < 1.]
            else:
                self.dataset = self.dataset[self.dataset.loc[:,3] > threshold]

            self.dataset.reset_index(drop=True, inplace=True)
            self.true_length = len(self.dataset)

        print("Length:", self.true_length, len(self), self.p, l_mean, flush=True)


    def __len__(self):
        return self.true_length if not self.max_size else self.max_size

    def __getitem__(self, idx):
        if self.max_size:
            idx = (idx + np.random.randint(0, self.true_length // self.max_size) * self.max_size)
        if self.pairs:
            idx1, idx2 = self.dataset.loc[idx,[0,1]]
            idx1, idx2 = int(idx1), int(idx2)
            img1 = Image.open(io.BytesIO(self.images[idx1])).convert("RGB")
            if random.random() < self.p:
                img2 = Image.open(io.BytesIO(self.images[idx2])).convert("RGB")
            else:
                img2 = img1
            return self.transform(img1, img2), idx



        while self.image_ids[idx] == -1:
            idx = random.randrange(len(self.images))
        img1 = Image.open(io.BytesIO(self.images[idx])).convert("RGB")

        # Default: random crop from the dataset
        # idx2 = random.randrange(len(self.images))

        # With probability p_same_image, sample from the same original image

        idx2 = idx
        if random.random() < self.p:
            candidates = self.pointer[int(self.image_ids[idx])]
            if len(candidates) > 1:
                while idx2 == idx:
                    idx2 = random.choice(candidates)
                img2 = Image.open(io.BytesIO(self.images[idx2])).convert("RGB")

        if idx2 == idx:
            img2 = img1

        return self.transform(img1, img2), idx