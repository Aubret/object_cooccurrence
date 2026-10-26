import os
from pathlib import Path

import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

# add knn args
parser.add_argument("--data_root", default="/work/autolearn/aubret/kitti", type=str)
# parse args
args = parser.parse_args()


calibrations = ["2011_09_26_drive_0119", "2011_09_28_drive_0225", "2011_09_29_drive_0108", "2011_09_30_drive_0072", "2011_10_03_drive_0058"]
person = "2011_09_28"

if __name__ == '__main__':
    dataset_root = Path(args.data_root)
    output_dir = Path(args.data_root) / "h5"
    output_dir.mkdir(exist_ok=True, parents=True)
    # class_wn_2_class_index = {k: i  for i, k in enumerate(os.listdir(dataset_root / "ecoset" /"test"))}
    with h5py.File(output_dir / f"kitti.h5", "w") as h5_file:
        mapper = []

        for d in ["2011_09_26", "2011_09_29", "2011_10_03", "2011_09_30"]
        image_filenames = []
        for dirpath in dataset_root.iterdir():
            if dirpath.is_dir():
                images = []
                # Sort to preserve consistent order (alphabetical)
                image_filenames = sorted([img for img in dirpath.glob("*/image_03/data/*.png")])

        print(len(image_filenames))
        # print(dataset_root / "ecoset" / split)
        cat_bar = tqdm(image_filenames, desc=f"dataset")

        dtype = h5py.vlen_dtype(np.dtype('uint8'))
        image_dataset = h5_file.create_dataset(f"images", shape=(len(image_filenames),), dtype=dtype)

        for i, image_filename in enumerate(cat_bar):
            image_dataset[i] = np.fromfile(image_filename, dtype=np.uint8)
            mapper.append(
                {
                    'h5_index': i,
                    "filename": image_filename.name,
                    "dirname": image_filename.parent.parent.parent.stem
                }
            )
        mapper = pd.DataFrame(mapper)
        # mapper['target'] = mapper['wn_name'].apply(lambda x: class_wn_2_class_index[x])
        # mapper['class_name'] = mapper['wn_name']
        #
        mapper.to_parquet(output_dir / f"kitti_mapper.parquet", index=False)