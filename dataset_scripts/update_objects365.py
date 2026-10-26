import h5py
import numpy as np
from PIL import Image
import io
from tqdm import tqdm

INPUT_H5 = "/work/autolearn/aubret/Objects365/train.h5"
OUTPUT_H5 = "/work/autolearn/aubret/Objects365/train_640.h5"

MAX_SIZE = 640
JPEG_QUALITY = 95


def resize_image(img):
    w, h = img.size

    if max(w, h) <= MAX_SIZE:
        return img

    scale = MAX_SIZE / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


with h5py.File(INPUT_H5, "r") as fin, h5py.File(OUTPUT_H5, "w") as fout:

    images_in = fin["images"]

    dt = h5py.vlen_dtype(np.dtype("uint8"))

    images_out = fout.create_dataset(
        "images",
        shape=(len(images_in),),
        dtype=dt,
    )

    for i in tqdm(range(len(images_in))):

        img_bytes = images_in[i]
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except:
            continue
        img = resize_image(img)

        buffer = io.BytesIO()
        img.save(
            buffer,
            format="JPEG",
            # quality=JPEG_QUALITY,
            # optimize=True,
        )

        images_out[i] = np.frombuffer(
            buffer.getvalue(),
            dtype=np.uint8,
        )