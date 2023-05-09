import argparse
import os
import pickle
from pathlib import Path

import gdown
import h5py
import numpy as np
from PIL import Image
from wilds import get_dataset


class FMoW:
    time_steps = list(range(16))
    drive_id = "1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3"
    file_name = "fmow.pkl"

    def __init__(self, time_step, split, data_dir):
        if time_step not in self.time_steps:
            raise ValueError(f"Unknown time `{time_step}. Choose a time_step in {self.time_steps}")
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self.dataset = datasets[time_step][split]
        del datasets
        self._root = get_dataset(dataset="fmow", root_dir=data_dir, download=True).root

    def __len__(self):
        return len(self.dataset["labels"])


def maybe_download(drive_id, destination_dir, destination_file_name):
    destination_dir = Path(destination_dir)
    destination = destination_dir / destination_file_name
    if destination.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(
        url=f"https://drive.google.com/u/0/uc?id={drive_id}&export=download&confirm=pbef",
        output=str(destination),
        quiet=False,
    )


def get_item(dataset, data_dir, idx):
    image_idx = dataset.dataset["image_idxs"][idx]
    img = Image.open(Path(data_dir) / "fmow_v1.1" / "images" / f"rgb_img_{image_idx}.png").convert(
        "RGB"
    )
    label = dataset.dataset["labels"][idx]
    return img, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="Folder where output file will be stored to.")
    args = parser.parse_args()
    splits = {"train": 0, "test": 1}
    with h5py.File(str(Path(args.data_dir) / "fmow.hdf5"), "w") as f:
        for time_step in FMoW.time_steps:
            time_key = str(time_step)
            f.create_group(time_key)
            for split in ["train", "test"]:
                f[time_key].create_group(split)
                d = FMoW(time_step=time_step, split=splits[split], data_dir=args.data_dir)
                f[time_key][split].create_dataset("images", (len(d), 224, 224, 3), dtype="u1")
                f[time_key][split].create_dataset("labels", (len(d),), dtype="u1")
                for i in range(len(d)):
                    data_image, data_label = get_item(d, args.data_dir, i)
                    f[time_key][split]["images"][i] = np.array(data_image)
                    f[time_key][split]["labels"][i] = int(data_label)
    print("File downloaded to", str(Path(args.data_dir) / "fmow.hdf5"))
