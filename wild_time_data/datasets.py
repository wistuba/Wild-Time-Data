import os
from abc import ABC, abstractmethod

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from wild_time_data.utils import maybe_download


class WildTimeDataset(Dataset, ABC):
    default_transform = None

    def __init__(
        self, time_step, split, data_dir, transform=None, target_transform=None, in_memory=False
    ):
        super().__init__()
        self.time_step = time_step
        self.split = split
        self.data_dir = data_dir
        self.in_memory = in_memory
        self._transform = transform or self.default_transform
        self._target_transform = target_transform
        self._file = None
        if time_step not in self.time_steps:
            raise ValueError(
                f"Unknown time_step `{time_step}. Choose a time_step in {self.time_steps}"
            )
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        time_step = str(time_step)
        if in_memory:
            with h5py.File(os.path.join(data_dir, self.file_name), "r") as f:
                self.X = self.load_x_in_memory(f[time_step][split])
                self.y = torch.from_numpy(f[time_step][split]["labels"][:])
        else:
            self._file = h5py.File(os.path.join(data_dir, self.file_name), "r")
            self.X = self.read_x_from_hdf5(self._file[time_step][split])
            self.y = self._file[time_step][split]["labels"]
        self.size = len(self.y)

    def __del__(self):
        if self._file:
            self._file.close()

    def __getitem__(self, idx):
        if isinstance(self.X, tuple):
            x = tuple(x_[idx] for x_ in self.X)
        else:
            x = self.X[idx]
        y = self.y[idx]
        if self._file:
            x = self._hdf5_transform(x)
            y = torch.tensor(y)
        if self._transform:
            x = self._transform(x)
        if self._target_transform:
            y = self._target_transform(y)
        return x, y

    def __len__(self):
        return self.size

    @abstractmethod
    def load_x_in_memory(self, data):
        pass

    @abstractmethod
    def read_x_from_hdf5(self, data):
        pass

    @staticmethod
    def _hdf5_transform(x):
        return x


class WildTimeImageDataset(WildTimeDataset, ABC):
    def load_x_in_memory(self, data):
        return [self._hdf5_transform(image) for image in data["images"]]

    def read_x_from_hdf5(self, data):
        return data["images"]


class WildTimeTextDataset(WildTimeDataset, ABC):
    @staticmethod
    def _hdf5_transform(x):
        return x.decode()

    def load_x_in_memory(self, data):
        return [self._hdf5_transform(text) for text in data["text"]]

    def read_x_from_hdf5(self, data):
        return data["text"]


class ArXiv(WildTimeTextDataset):
    time_steps = list(range(2007, 2023))
    input_dim = 55
    num_outputs = 172
    drive_id = "1dlYnA-C_4aHRLcjwNhrz768VkQ7E5Jn6"
    file_name = "arxiv.hdf5"
    checksum = "b6f2939f550188619f3b5850496909e81b89b132762fc3d6079d7f194acb0181c29a119009677221eaac25c234f962394a1ad125cb9e6f1d71ee23889623b1e7"  # noqa: E501


class Drug(WildTimeDataset):
    time_steps = list(range(2013, 2021))
    input_dim = [(63, 100), (26, 1000)]
    num_outputs = 1
    drive_id = "1mXCJ7zvgwRSl7XrKV0wU8GbVvZ3EHVTa"
    file_name = "drug.hdf5"
    mapping_amino = {key: value for value, key in enumerate("?ABCDEFGHIKLMNOPQRSTUVWXYZ")}
    mapping_smiles = {
        key: value
        for value, key in enumerate(
            "#%()+-.0123456789=?ABCDEFGHIKLMNOPRSTUVWYZ[]_abcdefghilmnorstuy"
        )
    }
    checksum = "df886f88d37592b503e6bab830478df8ebb9790f41272b413ad1972012647152d5fe823d36a75cadadb905ce4842aeeea14d8a40670734f0e002d39a168bc312"  # noqa: E501

    @staticmethod
    def default_transform(x):
        return Drug._transform_drug(x[0]), Drug._transform_protein(x[1])

    @staticmethod
    def _hdf5_transform(x):
        return x[0].decode(), x[1].decode()

    @staticmethod
    def _transform_drug(x):
        tensor = torch.zeros((63, 100))
        for i, c in enumerate(x):
            tensor[Drug.mapping_smiles[c], i] = 1
        return tensor

    @staticmethod
    def _transform_protein(x):
        tensor = torch.zeros((26, 1000))
        for i, c in enumerate(x):
            tensor[Drug.mapping_amino[c], i] = 1
        return tensor

    def load_x_in_memory(self, data):
        return [
            self._hdf5_transform((drug, protein))
            for drug, protein in zip(data["drug"], data["protein"])
        ]

    def read_x_from_hdf5(self, data):
        return data["drug"], data["protein"]


class FMoW(WildTimeImageDataset):
    time_steps = list(range(16))
    input_dim = (3, 224, 224)
    num_outputs = 62
    drive_id = "1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3"
    file_name = "fmow.hdf5"
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    checksum = "337de8819ccd2c7d74f70a97451db98ef012707f1cce6c583a4b9c5ee5fc90828b75f9b9eeb60e472c8a0bc12dda7f17f4f6dd8413295fa764f1293ba5c27af6"  # noqa: E501


class HuffPost(WildTimeTextDataset):
    time_steps = list(range(2012, 2019))
    input_dim = 44
    num_outputs = 11
    drive_id = "1kcILM6jENF_g2-0c5SkGsdXBGDkPjCHZ"
    file_name = "huffpost.hdf5"
    checksum = "e5bc15067eb2aedfe99c658b7b7016b5ad7feb3ce321be95545fdeda72e745296d61f741953940a217d4d9f582ffad28b3857b37719b833848654ecfbb94116c"  # noqa: E501


class Yearbook(WildTimeImageDataset):
    time_steps = list(range(1930, 2014))
    input_dim = (1, 32, 32)
    num_outputs = 2
    drive_id = "16lPT5DS3tz0XWnBuP8C-8zBnK0bwK7eX"
    file_name = "yearbook.hdf5"
    default_transform = transforms.ToTensor()
    checksum = "5199f73c0935b9ae874e0b5bfd444379a7993d7b58b948d591925e42a0dfdd84db6e940ff5e42409afd682da986feac88fe66e37d5c04dcfd3bec9c8b5354243"  # noqa: E501
