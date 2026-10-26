import csv
import io
import os
import random

import scipy
import h5py
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image


class MVImgNet(Dataset):
    def __init__(self, data_root, transform, split,  subset=0.1, time_window=30, action_label=False, action_padding=0, **kwargs):
        super().__init__()
        self.time_window = time_window
        self.split = split
        self.transform = transform
        self.action_label = action_label
        self.action_padding = action_padding

        self.category_mapping = {}
        category_list = []
        for index, v in pd.read_csv(os.path.join(data_root, "mvimgnet_category.txt"), header=None).iterrows():
            self.category_mapping[v[0]] = index
            category_list.append(v[0])
        self.n_classes = len(self.category_mapping)
        if not os.path.exists(os.path.join(data_root, "data_all.h5")):
            raise Exception("You should build the merged data hdf5 file")
        self.hdf5_file = h5py.File(os.path.join(data_root, "data_all.h5"), "r")
        self.dataset = pd.read_parquet(os.path.join(data_root, f"dataset_{self.split}_all3.parquet"))

        if subset and self.split == "train":
            uniqs = self.dataset["object"].unique()
            pathobj = os.path.join(data_root, f"subset_0_{subset}")
            if not os.path.exists(pathobj):
                sampled_objects = np.random.choice(uniqs, int(len(uniqs)*subset))
                f = open(pathobj, "w")
                csv.writer(f).writerow(sampled_objects)
                f.close()
            with open(pathobj, "r") as f:
                sampled_objects = next(iter(csv.reader(f)))
            self.dataset = self.dataset.query('object in @sampled_objects').reset_index(drop=True)

        # self.action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723 , 0.00314, 0.00787, -0.0171])
 #        self.action_mean = torch.tensor([0.00459311,  0.09484751,  0.15115935,  0.7198106, -0.02323676, -0.02354124,
 #  0.02108987])
 #        # self.action_std = torch.tensor([0.431, 0.048, 0.358, 0.328, 3.8, 1.25, 1.13])
 #        self.action_std = torch.tensor([0.03695753, 0.31131427, 0.40189099, 0.4361283, 0.22864199, 0.03714998,
 # ,0.04096671])
        print(len(self.dataset["object"].unique()))
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def open_image(self, category, obj, index, path, idx):
        partition = self.dataset.loc[idx, "partition"]
        p = self.hdf5_file.get(partition)
        c = p.get(category)
        o = c.get(obj)
        return Image.open(io.BytesIO(o[index]))

    def get_action(self, idx, new_idx, image):
        r1 = self.dataset.loc[idx, ["q1", "q2", "q3","q0"]].values
        r2 = self.dataset.loc[new_idx, ["q1", "q2", "q3","q0"]].values

        t1 = self.dataset.loc[idx, ["t0", "t1", "t2"]].values
        t2 = self.dataset.loc[new_idx, ["t0", "t1", "t2"]].values

        R1 = scipy.spatial.transform.Rotation.from_quat(r1).as_matrix()
        R2 = scipy.spatial.transform.Rotation.from_quat(r2).as_matrix()

        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1

        # R_rel = R1.T @ R2
        # t_rel = R1.T @ (t2 - t1)
        r_rel = scipy.spatial.transform.Rotation.from_matrix(R_rel).as_quat()

        action = torch.tensor(np.concatenate((
            r_rel.astype(np.float32),
            t_rel.astype(np.float32)), axis=0
        ))

        action[4:] = action[4:] / (torch.norm(action[4:]) + 1e-6)
        # action = (action - self.action_mean)/self.action_std
        # if torch.norm(action, dim=0) > 50:
        #     print(action)
        # action = torch.clamp(action, -5, 5)
        if self.action_padding:
            action = torch.cat((torch.zeros((2,)), action))

        return action

    def __getitem__(self, idx):
        # idx=530
        category, obj, frame_index = self.dataset.loc[idx, "category"],self.dataset.loc[idx, "object"],self.dataset.loc[idx, "frame"]
        size = self.dataset.loc[idx, "length"]
        image = self.open_image(str(category), str(obj),frame_index, self.dataset.loc[idx, "path"], idx)

        label = np.int64(self.category_mapping[category])

        if not self.time_window:
            # return self.transform(image), label
            return self.transform(image, image), label

        new_frame_index = (frame_index + random.randint(-self.time_window, self.time_window))%size
        new_idx = idx + new_frame_index - frame_index

        assert str(obj) == str(self.dataset.loc[new_idx, "object"]), f"{str(obj)}, {str(self.dataset.loc[new_idx, 'object'])}, {idx}, {new_idx}, {frame_index}, {new_frame_index}, {size}"
        image_pair = self.open_image(str(category), str(obj), self.dataset.loc[new_idx, "frame"],self.dataset.loc[new_idx, "path"], new_idx)

        action = self.get_action(idx, new_idx, image)

        if self.action_label:
            return self.transform(image), self.transform(image_pair), action
        return self.transform(image, image_pair, None, None, action), label