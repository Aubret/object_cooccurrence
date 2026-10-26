import hashlib
import io
import json
import os
import random
from pathlib import Path
from typing import Union, Callable, Optional, Tuple
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from solo.data.custom.base import H5ClassificationDataset



class Ecoset(H5ClassificationDataset):
    _SPLITS = ["train", "val", "test"]
    _H5_FILENAME = "ecoset_{split}.h5"
    _MAPPER_FILENAME = "ecoset_{split}_mapper.parquet"
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)


class ImgnetDataset(Dataset):

    def __init__(self, data_root, split, transform, imgnet100=True, percent=100):
        # super().__init__()
        self.mode = "train" if split == "train" else "val"
        # driver = "core" if self.mode == "train" and (imgnet100 or percent != 100) else None
        if self.mode == "val":
            self.hdf5_file = h5py.File(os.path.join(data_root, f'data2_{self.mode}.h5'),"r", driver=None)
        else:
            self.hdf5_file = h5py.File(os.path.join(data_root, "..", f'train.h5'),"r", driver=None)
        self.dataset = pd.read_csv(os.path.join(data_root, f'dataset2_{self.mode}.csv'), header=None)
        self.dataset.columns.astype(str)
        self.dataset.columns = ["index","0", "1", "2", "label", "4"] if self.mode == "train" else ["0", "1", "2", "label", "4"]
        self.transform = transform
        self.n_classes = 1000
        self.imgnet100 = imgnet100

        if percent != 100 and split == "train":
            id_to_num = json.load(open(os.path.join(data_root, "ids.json"), "r"))
            id_list = open(os.path.join(data_root, f"{str(percent)}percent.txt"), "r")
            num_list = [id_to_num[i] for i in id_list.read().splitlines()]
            self.dataset = self.dataset.iloc[num_list].reset_index(drop=True)

        if imgnet100:
            self.n_classes = 100
            name_id_map = json.load(open(os.path.join(data_root, "imagenet_class_index.json"), "r"))
            mapping = {v[0]: int(k) for k, v in name_id_map.items()}

            classes_file = "solo/data/dataset_subset/imagenet100_classes.txt"
            with open(classes_file) as f:
                self.classes = f.readline().strip().split()
            self.id_classes = sorted(self.classes)

            catfilter = list(map(lambda x: mapping[x], self.id_classes))
            self.category_mapping = {cat: i for i, cat in enumerate(catfilter)}

            self.dataset = self.dataset.query("label in @catfilter")
            self.dataset = self.dataset.reset_index(drop=True)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        h5_index, label, begin = self.dataset.loc[idx, "0"], self.dataset.loc[idx, "label"], self.dataset.loc[
            idx, '4']
        # img = Image.open("../datasets/imgnet/train/ILSVRC2012_val_00012562_n07697313.JPEG")
        if self.mode == "val":
            img = Image.open(io.BytesIO(self.hdf5_file.get(f"data2_{begin}")[h5_index]))
        else:
            img = Image.open(io.BytesIO(self.hdf5_file.get("images")[h5_index]))
            # h5_index = h5_index%50000
            # img = Image.open(io.BytesIO(self.hdf5_file.get(str(begin)).get("data")[h5_index]))
        if self.imgnet100:
            label = self.category_mapping[label]
        x = self.transform(img)
        return x, label

class ImgnetContextDataset(Dataset):

    def __init__(self, data_root, split, transform, p=0, n_context=100, seed=0, one_assignment=0, freq_inter=1, mapping=False,**kwargs):
        # super().__init__()
        self.mode = "train" if split == "train" else "val"
        # driver = "core" if self.mode == "train" and (imgnet100 or percent != 100) else None
        if self.mode == "val":
            self.hdf5_file = h5py.File(os.path.join(data_root, f'data2_{self.mode}.h5'),"r", driver=None)
        else:
            self.hdf5_file = h5py.File(os.path.join(data_root, "..", f'train.h5'),"r", driver=None)
        self.dataset = pd.read_csv(os.path.join(data_root, f'dataset2_{self.mode}.csv'), header=None)
        self.dataset.columns.astype(str)
        self.dataset.columns = ["index","0", "1", "2", "label", "4"] if self.mode == "train" else ["0", "1", "2", "label", "4"]
        self.dataset["index"] = self.dataset.index
        self.transform = transform
        self.n_classes = 1000
        self.one_assignment = one_assignment
        self.freq_inter = freq_inter
        self.mapping = mapping
        # self.dataset["label"]
        # .drop_duplicates()
        df_shuffle = (
            pd.DataFrame(np.arange(1000))
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        groups = np.array_split(df_shuffle.values, n_context)
        label_to_context = {
            label[0]: i
            for i, group in enumerate(groups)
            for label in group
        }
        self.dataset["context"] = self.dataset["label"].map(label_to_context)
        context_groups = self.dataset.groupby('context')
        self.category_to_labels = context_groups['label'].unique().apply(list).to_dict()
        self.category_to_ids = context_groups['index'].unique().apply(list).to_dict()
        # for context, ids in self.category_to_ids.items():
        #     print(f"Category {context} has {len(ids)} IDs.")
        # seed_new = random.choice(seeds) % len(self.category_to_ids[context])
        # self.dataset = self.dataset.
        self.p = p
        self.split = split
        np.random.seed(seed+2)
        self.chosen = np.random.rand(len(self)) < self.freq_inter


        if self.mapping:
            partners = np.arange(len(self), dtype=int)
            for context, ids in self.category_to_ids.items():

                n = len(ids)
                perm = np.array(ids)[np.random.permutation(n)]

                for i in range(0, n//2):
                    if random.random() < self.freq_inter:
                        a = perm[i]
                        b = perm[i + n//2]

                        partners[a] = b
                        partners[b] = a

            self.dataset["partners"] = partners
            print(partners)
        print(df_shuffle)
        print(self.chosen)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        h5_index, label, begin, context = self.dataset.loc[idx, ["0", "label", '4',"context"]]
        # img = Image.open("../datasets/imgnet/train/ILSVRC2012_val_00012562_n07697313.JPEG")
        if self.mode == "val":
            img = Image.open(io.BytesIO(self.hdf5_file.get(f"data2_{begin}")[h5_index]))
        else:
            img = Image.open(io.BytesIO(self.hdf5_file.get("images")[h5_index]))
            # h5_index = h5_index%50000
            # img = Image.open(io.BytesIO(self.hdf5_file.get(str(begin)).get("data")[h5_index]))
        if self.split == "train" and random.random() < self.p:
            if self.mapping:
                idx_new = self.dataset.loc[idx, "partners"]
                h5_index_new, label_new, begin_new, context_new = self.dataset.loc[idx_new, ["0", "label", '4', "context"]]
                if label_new == label:
                    img_new = img
                else:
                    img_new = Image.open(io.BytesIO(self.hdf5_file.get("images")[h5_index_new]))

            elif self.one_assignment:
                # if idx %self.freq_inter == 0:
                if self.chosen[idx]:
                    # idx_news = (idx * 1664525 + 1013904223)%len(self.category_to_ids[context])
                    seeds = [((idx * 73856093) ^ (i * 19349663)) & 0xFFFFFFFF for i in range(self.one_assignment)]
                    seed_new = random.choice(seeds)%len(self.category_to_ids[context])
                    idx_new = self.category_to_ids[context][seed_new]
                    h5_index_new, label_new, begin_new, context_new = self.dataset.loc[idx_new, ["0", "label", '4', "context"]]
                    if label_new == label:
                        img_new = img
                    else:
                        img_new = Image.open(io.BytesIO(self.hdf5_file.get("images")[h5_index_new]))
                else:
                    img_new = img
            else:
                idx_new = random.choice(self.category_to_ids[context])
                h5_index_new, label_new, begin_new, context_new = self.dataset.loc[idx_new, ["0", "label", '4', "context"]]
                while label_new == label:
                    idx_new = random.choice(self.category_to_ids[context])
                    h5_index_new, label_new, begin_new, context_new = self.dataset.loc[idx_new, ["0", "label", '4', "context"]]
                img_new = Image.open(io.BytesIO(self.hdf5_file.get("images")[h5_index_new]))
            x = self.transform(img, img_new)
        else:
            x = self.transform(img)
        return x, label


class ImgNetDataset_42(H5ClassificationDataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            subset: Optional[str] = None
    ):
        # super().__init__(root, transform, split, driver="core" if split == "train" and subset in ["1pct", "10pct", "imgnet100"] else None)
        super().__init__(root, transform, split, driver=None)
        if subset == '1pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_1percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 1%")
        elif subset == '10pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_10percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 10%")
        elif subset in ["imgnet100", "imgnet100_im"]:
            with open(self.root / 'imagenet100_classes.txt') as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {class_wn: class_index for class_index, class_wn in
                                                enumerate(imgnet100_classes)}
            self.mapper = self.mapper.query('wn_name in @imgnet100_classes').reset_index(drop=True)
            self.mapper['target'] = self.mapper['wn_name'].apply(lambda x: imgnet100_class_wn_2_class_index[x])

        self.n_classes = self.mapper['target'].nunique()
        self.target_2_class_name = self.mapper[['target', 'class_name']].drop_duplicates().set_index('target')[
            'class_name'].to_dict()

class ImgNetDatasetContext_42(H5ClassificationDataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            subset: Optional[str] = None,
            p=0.1,
            n_context=100,
            seed=0
    ):
        super().__init__(root, transform, split, driver="core" if split == "train" and subset in ["1pct", "10pct", "imgnet100"] else None)
        if subset == '1pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_1percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 1%")
        elif subset == '10pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_10percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 10%")
        elif subset in ["imgnet100", "imgnet100_im"]:
            with open(self.root / 'imagenet100_classes.txt') as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {class_wn: class_index for class_index, class_wn in
                                                enumerate(imgnet100_classes)}
            self.mapper = self.mapper.query('wn_name in @imgnet100_classes').reset_index(drop=True)
            self.mapper['target'] = self.mapper['wn_name'].apply(lambda x: imgnet100_class_wn_2_class_index[x])

        self.n_classes = self.mapper['target'].nunique()
        self.target_2_class_name = self.mapper[['target', 'class_name']].drop_duplicates().set_index('target')[
            'class_name'].to_dict()
        self.mapper["index"] = self.mapper.index

        df_shuffle = (
            pd.DataFrame(np.arange(1000))
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        groups = np.array_split(df_shuffle.values, n_context)
        label_to_context = {
            label[0]: i
            for i, group in enumerate(groups)
            for label in group
        }
        self.mapper["context"] = self.mapper["target"].map(label_to_context)
        context_groups = self.mapper.groupby('context')
        self.category_to_labels = context_groups['target'].unique().apply(list).to_dict()
        self.category_to_ids = context_groups['index'].unique().apply(list).to_dict()
        # for context, ids in self.category_to_ids.items():
        #     print(f"Category {context} has {len(ids)} IDs.")
        self.p = p
        self.split = split


    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        # if self.mapper is None:
        #     raw_image = self.h5_file.get("images")[idx]
        #     target = self.h5_file.get("targets")[idx]
        # else:
        dp = self.mapper.loc[idx]
        raw_image = self.h5_file.get("images")[dp["h5_index"]]
        target, context = dp[["target","context"]]

        image = Image.open(io.BytesIO(raw_image)).convert("RGB")

        if self.split == "train" and random.random() < self.p:
            idx_new = random.choice(self.category_to_ids[context])
            label_new, context_new, h5_index_new = self.mapper.loc[idx_new, ["target", "context","h5_index"]]
            while label_new == target:
                idx_new = random.choice(self.category_to_ids[context])
                label_new, context_new, h5_index_new = self.mapper.loc[idx_new, ["target", "context","h5_index"]]
            assert context_new == context
            img_new = Image.open(io.BytesIO(self.h5_file.get("images")[h5_index_new])).convert("RGB")
            image = self.transform(image, img_new)
        else:
            image = self.transform(image)


        return image, target


class ImageNetOODDataset(Dataset):
    VERSION = ['colour',
               'contrast',
               'cue-conflict',
               'edge',
               'eidolonI',
               'eidolonII',
               'eidolonIII',
               'false-colour',
               'high-pass',
               'low-pass',
               'phase-scrambling',
               'power-equalisation',
               'rotation',
               'silhouette',
               'sketch',
               'stylized',
               'uniform-noise']

    def __init__(self,
                 path: str,
                 version: str,
                 return_classname: bool = False,
                 transform: Optional[Callable] = None):
        self.version = version
        self.transform = transform
        self.return_classname = return_classname
        self.h5_file = h5py.File(path, "r", driver="core")

        self.available_versions = list(self.h5_file.keys())
        if not self.version in self.available_versions:
            raise ValueError(f"Version {self.version} not available, choose one of {self.available_versions}")

    def __len__(self) -> int:
        return self.h5_file.get(self.version).get('targets').shape[0]

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Union[str, np.ndarray]]:
        version = self.h5_file.get(self.version)
        image = Image.fromarray(version.get("images")[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_classname:
            targets = version.get("classnames")[idx]
        else:
            targets = version.get("targets")[idx]

        return image, targets


class ImageNetS(ImageFolder):
    def __init__(self, root, transform=None, mode: str = 'full'):
        super().__init__(root, transform)
        mask_p = root + '-segmentation/'
        self.masks = [Path(mask_p + '/'.join(x[0].split('/')[-2:])).with_suffix('.png') for x in self.samples]
        self.mode = mode
        if self.mode not in ['full', 'fg', 'bg', 'bg_rec']:
            raise ValueError(f"Mode {self.mode} not available, choose one of ['full', 'fg', 'bg', 'bg_rec']")

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.mode == "fg":
            mask = np.array(Image.open(self.masks[index]))[..., 0] > 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == "bg":
            mask = 1 - (np.array(Image.open(self.masks[index]))[..., 0] > 0).astype(np.uint8)
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == 'bg_rec':
            mask = np.array(Image.open(self.masks[index]))[..., 0]
            bbox = cv2.boundingRect(mask)
            mask = np.ones_like(mask)
            mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
