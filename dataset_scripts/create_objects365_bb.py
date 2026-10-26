import io
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/work/autolearn/aubret/Objects365/"

INPUT_H5 = f"{DATA_ROOT}/train.h5"
OUTPUT_H5 = f"{DATA_ROOT}/objects365_crops.h5"

MIN_SIZE = 10000
JPEG_QUALITY = 95

# ---------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------

mapping = json.load(open(f"{DATA_ROOT}/mapping.json"))
dataset = json.load(open(f"{DATA_ROOT}/zhiyuan_objv2_train.json"))

image_dict = {im["id"]: im for im in dataset["images"]}

rows = []

for ann in dataset["annotations"]:

    if str(ann["image_id"]) not in mapping:
        continue

    img = image_dict[ann["image_id"]]

    x = ann["bbox"][0]
    y = ann["bbox"][1]
    w = ann["bbox"][2]
    h = ann["bbox"][3]

    if w * h < MIN_SIZE:
        continue

    rows.append({
        "img_idx": mapping[str(ann["image_id"])],
        "image_id": ann["image_id"],
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    })

df = pd.DataFrame(rows)

print(len(df), "bounding boxes")

# ---------------------------------------------------------------------
# Create output HDF5
# ---------------------------------------------------------------------

with h5py.File(INPUT_H5, "r") as fin, \
     h5py.File(OUTPUT_H5, "w") as fout:

    images_in = fin["images"]

    dt = h5py.vlen_dtype(np.dtype("uint8"))

    images_out = fout.create_dataset(
        "images",
        shape=(len(df),),
        dtype=dt,
    )

    image_ids = fout.create_dataset(
        "image_ids",
        shape=(len(df),),
        dtype=np.int32,
    )

    for i, row in tqdm(df.iterrows(), total=len(df)):

        img_idx = int(row.img_idx)

        img = Image.open(
            io.BytesIO(images_in[img_idx])
        ).convert("RGB")

        W, H = img.size

        x1 = max(0, row.x)
        y1 = max(0, row.y)
        x2 = min(W, row.x + row.w)
        y2 = min(H, row.y + row.h)
        if x2 < x1 or y2 < y1:
            image_ids[i] = -1
            continue
        crop = img.crop((x1, y1, x2, y2))

        buffer = io.BytesIO()
        crop.save(
            buffer,
            format="JPEG"
        )

        images_out[i] = np.frombuffer(
            buffer.getvalue(),
            dtype=np.uint8,
        )

        image_ids[i] = row.image_id