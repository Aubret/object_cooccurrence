import io
from pathlib import Path
from typing import Union, Callable, Optional, Tuple

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset




class H5ClassificationDataset(Dataset):
    _SPLITS = ["train", "val", "test"]
    _H5_FILENAME = "{split}.h5"
    _MAPPER_FILENAME = "{split}_mapper.parquet"

    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            driver: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        assert self.split in self._SPLITS

        h5_path = self.root / self._H5_FILENAME.format(split=self.split)
        if not h5_path.exists():
            raise FileNotFoundError(f"{h5_path} does not exists.")
        self.h5_file = h5py.File(h5_path, "r", driver=driver)

        self.mapper = None
        if "targets" not in self.h5_file.keys():
            self.mapper = pd.read_parquet(self.root / self._MAPPER_FILENAME.format(split=self.split))
        print("dataset", len(self))

    def __len__(self) -> int:
        return len(self.h5_file.get("images")) if self.mapper is None else len(self.mapper)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        if self.mapper is None:
            raw_image = self.h5_file.get("images")[idx]
            target = self.h5_file.get("targets")[idx]
        else:
            dp = self.mapper.loc[idx]
            raw_image = self.h5_file.get("images")[dp["h5_index"]]
            target = dp["target"]

        image = Image.open(io.BytesIO(raw_image)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target
