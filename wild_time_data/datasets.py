import os
import pickle
from abc import ABC, abstractmethod

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from wilds import get_dataset

from wild_time_data.utils import embed_drug, embed_protein, maybe_download


class WildTimeDataset(Dataset, ABC):
    def __init__(self, time_step, split, data_dir):
        super().__init__()
        if time_step not in self.time_steps:
            raise ValueError(f"Unknown time `{time_step}. Choose a time_step in {self.time_steps}")
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets[time_step][split]
        del datasets

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self._dataset["labels"])


class WildTimeImageDataset(WildTimeDataset, ABC):
    def __len__(self):
        return len(self._dataset["labels"])


class WildTimeTextDataset(WildTimeDataset, ABC):
    def __len__(self):
        return len(self._dataset["category"])


class ArXiv(WildTimeTextDataset):
    time_steps = [i for i in range(2007, 2023)]
    input_dim = 55
    num_classes = 172
    drive_id = "1H5xzHHgXl8GOMonkb6ojye-Y2yIp436V"
    file_name = "arxiv.pkl"

    def __getitem__(self, idx):
        return self._dataset["title"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]

    def __len__(self):
        return len(self._dataset["category"])


class Drug(WildTimeDataset):
    time_steps = [i for i in range(2013, 2021)]
    input_dim = [(63, 100), (26, 1000)]
    num_classes = 1
    drive_id = "12SmQXA6f1fPd9__WAY8lravVAlDsFP7p"
    file_name = "drug.pkl"

    def __getitem__(self, index):
        return (
            embed_drug(self._dataset.iloc[index].Drug_Enc),
            embed_protein(self._dataset.iloc[index].Target_Enc),
        ), self._dataset.iloc[index].Y

    def __len__(self):
        return len(self._dataset)


class FMoW(WildTimeImageDataset):
    time_steps = [i for i in range(16)]
    input_dim = (3, 32, 32)
    num_classes = 62
    drive_id = "1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3"
    file_name = "fmow.pkl"

    def __init__(self, time_step, split, data_dir):
        super().__init__(time_step, split, data_dir)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._root = get_dataset(dataset="fmow", root_dir=data_dir, download=True).root

    def __getitem__(self, idx):
        idx = self._dataset["image_idxs"][idx]
        img = Image.open(self._root / "images" / f"rgb_img_{idx}.png").convert("RGB")
        image = self._transform(img)
        label = torch.LongTensor([self._dataset["labels"][idx]])[0]
        return image, label


class HuffPost(WildTimeTextDataset):
    time_steps = [i for i in range(2012, 2019)]
    input_dim = 44
    num_classes = 11
    drive_id = "1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY"
    file_name = "huffpost.pkl"

    def __getitem__(self, idx):
        return self._dataset["headline"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]


class Yearbook(WildTimeImageDataset):
    time_steps = [i for i in range(1930, 2014)]
    input_dim = (3, 32, 32)
    num_classes = 2
    drive_id = "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"
    file_name = "yearbook.pkl"

    def __getitem__(self, idx):
        image = torch.FloatTensor(self._dataset["images"][idx]).permute(2, 0, 1)
        label = torch.LongTensor([self._dataset["labels"][idx]])[0]
        return image, label
