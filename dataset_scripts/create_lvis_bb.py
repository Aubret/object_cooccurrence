import os
import io
import h5py
import pandas as pd
import numpy as np

from PIL import Image
from lvis import LVIS
from tqdm import tqdm


def precrop_lvis(
    data_root,
    output_path,
    split="train",
    min_size=1,
    scale=0.5,
    jpeg_quality=95,
):
    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------
    lvis_path = os.path.join(data_root,f"lvis_v1_{split}.json")

    csv_path = os.path.join(data_root,f"dataset{'' if split == 'train' else '_val'}.csv")
    image_h5_path = os.path.join(data_root,f"lvis{'' if split == 'train' else '_val'}.h5")

    lvis_api = LVIS(lvis_path)

    df = pd.read_csv(csv_path, header=None, names=[
            "x", "y", "w", "h",
            "cat_name",
            "img_idx",
            "img_file",
            "coco_url"
        ]
    )

    # Category IDs
    cat_to_id = {
        cat["name"]: cat["id"]
        for cat in lvis_api.load_cats(
            lvis_api.get_cat_ids()
        )
    }

    df["cat_id"] = df["cat_name"].map(cat_to_id)

    # Same filtering as your dataset
    df = df[(df["w"] * df["h"]) > min_size]
    df = df.reset_index(drop=True)

    print(f"Number of crops: {len(df)}")

    # ---------------------------------------------------------
    # Open source HDF5
    # ---------------------------------------------------------
    src_h5 = h5py.File(image_h5_path, "r")
    images = src_h5["images"]

    # ---------------------------------------------------------
    # Output HDF5
    # ---------------------------------------------------------
    out_h5 = h5py.File(output_path, "w")

    # Variable-length uint8 arrays
    jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))

    crops = out_h5.create_dataset(
        "images",
        shape=(len(df),),
        dtype=jpeg_dtype
    )

    cat_ids = out_h5.create_dataset(
        "cat_id",
        shape=(len(df),),
        dtype="int64"
    )

    img_idx_ds = out_h5.create_dataset(
        "img_idx",
        shape=(len(df),),
        dtype="int64"
    )

    # Store original row index
    original_idx = out_h5.create_dataset(
        "original_idx",
        shape=(len(df),),
        dtype="int64"
    )

    size = out_h5.create_dataset(
        "size",
        shape=(len(df),),
        dtype="int64"
    )

    # ---------------------------------------------------------
    # Cache decoded full images
    # ---------------------------------------------------------
    current_img_idx = None
    current_img = None

    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Precaching {split}"
    ):

        img_idx = int(row["img_idx"])

        # Only decode each original image once
        if img_idx != current_img_idx:
            current_img = Image.open(io.BytesIO(images[img_idx])).convert("RGB")
            current_img_idx = img_idx

        img = current_img

        # -----------------------------------------------------
        # Compute expanded crop
        # Same logic as your get_images()
        # -----------------------------------------------------
        x = int(row["x"] - scale * row["w"] // 2)
        y = int(row["y"] - scale * row["h"] // 2)
        w = int(scale * row["w"] + row["w"])
        h = int(scale * row["h"] + row["h"])

        width, height = img.size

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = max(x1 + 1, min(width, x + w))
        y2 = max(y1 + 1, min(height, y + h))

        crop = img.crop((x1, y1, x2, y2))

        # -----------------------------------------------------
        # JPEG encode
        # -----------------------------------------------------
        buffer = io.BytesIO()

        crop.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality
        )

        crops[i] = np.frombuffer(
            buffer.getvalue(),
            dtype=np.uint8
        )

        cat_ids[i] = int(row["cat_id"])
        img_idx_ds[i] = img_idx
        original_idx[i] = i
        size[i] = row["w"] * row["h"]

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------
    df.to_csv(
        output_path.replace(".h5", ".csv"),
        index=False
    )

    out_h5.close()
    src_h5.close()

    print(f"Saved: {output_path}")

precrop_lvis("/work/autolearn/aubret/coco/","/work/autolearn/aubret/coco/lvis_bb_train_scale05_min500.h5" ,split="train",min_size=500)
precrop_lvis("/work/autolearn/aubret/coco/","/work/autolearn/aubret/coco/lvis_bb_val_scale05_min500.h5" ,split="val",min_size=500)
