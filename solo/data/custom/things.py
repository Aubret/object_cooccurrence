from pathlib import Path
from typing import Union

import PIL
import pandas as pd
from torch.utils.data import Dataset


class Things(Dataset):
    def __init__(self, root: Union[str, Path], transforms, first_only=False, classes=None):
        self.root = root #/ "object_images"
        self.csv_file = pd.read_csv(root / "image-paths.csv", names=["image"], header=None)
        self.csv_file['nameID'] = self.csv_file['image'].apply(lambda x: x.split('/')[1])


        self.codes = pd.read_csv(root / "unique_id.txt", names=["names"], header=None)['names'].to_list()
        mapping = {name: i for i, name in enumerate(self.codes)}

        self.csv_file['ID'] = self.csv_file['nameID'].apply(lambda x: mapping[x])
        #self.csv_file.sort_values(by=["ID", 'image'])
        self.transforms = transforms
        if first_only:
            self.csv_file = self.csv_file.groupby('ID').first().reset_index()

        if classes:
            self.csv_file = self.csv_file[self.csv_file["nameID"].isin(classes)].reset_index(drop=True)
            mapping = {name: i for i, name in enumerate(classes)}
            self.csv_file['ID'] = self.csv_file['nameID'].apply(lambda x: mapping[x])
            self.codes = classes

        self.mapping = mapping
        # print(len(self))

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        path, concept_id = self.csv_file.loc[idx, ["image","ID"]]

        return self.transforms(PIL.Image.open(self.root / path)), int(concept_id)