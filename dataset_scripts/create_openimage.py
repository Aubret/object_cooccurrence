import io
import os

import PIL
from PIL import Image
import numpy as np
import pandas
import h5py

detection_path = "/scratch_shared/aubret/openimages/open-images-v7/train/labels/detections.csv"
csv_file = pandas.read_csv(detection_path)
#X = ["XClick1X","XClick2X","XClick3X","XClick4X"]
#Y = ["XClick1Y","XClick2Y","XClick3Y","XClick4Y"]

# print(csv_file.groupby("Source").groups.keys())

with h5py.File("/scratch_shared/aubret/openimages/openimages.h5", "w") as f:
    # dataset = f.create_dataset("images", shape=(len(csv_file),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    dataset = f.create_dataset("images", shape=(300,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    previous = "none"
    for i,  r in csv_file.iterrows():
        if previous != r['ImageID']:
            full_img = PIL.Image.open(f"/scratch_shared/aubret/openimages/open-images-v7/train/data/{r['ImageID']}.jpg")

        w, h = full_img.size
        img = full_img.crop((r["XMin"] *w, r["YMin"] *h, r["XMax"] *w , r["YMax"] *h))
        # img2 = full_img.crop((x_left, y_down, x_right, y_up))
        # print(img.size, x_left, x_right, y_up, y_down)

        # dataset[i] = np.fromfile(os.path.join("/scratch_shared/aubret/openimages/open-images-v7/train/data",f"{r["ImageID"]}.jpg"), dtype=np.uint8)
        os.makedirs(f"/scratch_shared/aubret/openimages/tests/{r['ImageID']}", exist_ok=True)
        img.save(f"/scratch_shared/aubret/openimages/tests/{r['ImageID']}/{i}.png")
        # img2.save(f"/scratch_shared/aubret/openimages/tests/{i}_1.png")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        saved_img = buffer.getvalue()  # returns raw bytes
        dataset[i] = np.frombuffer(saved_img, dtype=np.uint8)

        previous = r['ImageID']
        if i >= 300-1:
            break