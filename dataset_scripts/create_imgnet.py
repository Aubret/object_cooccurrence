from pathlib import Path

import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

# add knn args
parser.add_argument("--data_root", default="/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/", type=str)
parser.add_argument("--data_dest", default="/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/", type=str)
# parse args
args = parser.parse_args()

if __name__ == '__main__':
    dataset_root = Path(args.data_root)
    output_dir = Path(args.data_dest) / "ImageNet" / "h5"
    output_dir.mkdir(exist_ok=True, parents=True)


    # get the official ImageNet class map and class names
    class_map_url = "https://raw.githubusercontent.com/raghakot/keras-vis/refs/heads/master/resources/imagenet_class_index.json"
    class_map = pd.read_json(class_map_url, orient="index").reset_index().rename(
        columns={"index": "class_index", 0: "class_wn", 1: "class_name"})
    class_map["class_index"] = class_map["class_index"].astype(int)
    assert len(class_map) == 1000
    assert set(class_map["class_index"]) == set(range(1000))

    class_wn_2_class_index = class_map.set_index("class_wn")["class_index"].to_dict()
    class_wn_2_class_name = class_map.set_index("class_wn")["class_name"].to_dict()

    for split in ["train", "val"]:
        with h5py.File(output_dir / f"ImageNet_{split}.h5", "w") as h5_file:
            mapper = []

            image_filenames = list((dataset_root / "ImageNet" / split).rglob("*.JPEG"))
            cat_bar = tqdm(image_filenames, desc=f"{split} split")

            dtype = h5py.vlen_dtype(np.dtype('uint8'))
            image_dataset = h5_file.create_dataset(f"images", shape=(len(image_filenames),), dtype=dtype)

            for i, image_filename in enumerate(cat_bar):
                image_dataset[i] = np.fromfile(image_filename, dtype=np.uint8)
                mapper.append(
                    {
                        'h5_index': i,
                        "filename": image_filename.name,
                        "wn_name": image_filename.parent.stem
                    }
                )
        mapper = pd.DataFrame(mapper)
        mapper['target'] = mapper['wn_name'].apply(lambda x: class_wn_2_class_index[x])
        mapper['class_name'] = mapper['wn_name'].apply(lambda x: class_wn_2_class_name[x])
        mapper.to_parquet(output_dir / f"ImageNet_{split}_mapper.parquet", index=False)