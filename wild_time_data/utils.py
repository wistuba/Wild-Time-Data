from pathlib import Path

import gdown
import numpy as np
from sklearn.preprocessing import OneHotEncoder

amino_char = [
    "?",
    "A",
    "C",
    "B",
    "E",
    "D",
    "G",
    "F",
    "I",
    "H",
    "K",
    "M",
    "L",
    "O",
    "N",
    "Q",
    "P",
    "S",
    "R",
    "U",
    "T",
    "W",
    "V",
    "Y",
    "X",
    "Z",
]

smiles_char = [
    "?",
    "#",
    "%",
    ")",
    "(",
    "+",
    "-",
    ".",
    "1",
    "0",
    "3",
    "2",
    "5",
    "4",
    "7",
    "6",
    "9",
    "8",
    "=",
    "A",
    "C",
    "B",
    "E",
    "D",
    "G",
    "F",
    "I",
    "H",
    "K",
    "M",
    "L",
    "O",
    "N",
    "P",
    "S",
    "R",
    "U",
    "T",
    "W",
    "V",
    "Y",
    "[",
    "Z",
    "]",
    "_",
    "a",
    "c",
    "b",
    "e",
    "d",
    "g",
    "f",
    "i",
    "h",
    "m",
    "l",
    "o",
    "n",
    "s",
    "r",
    "u",
    "t",
    "y",
]


enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))


def embed_protein(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T


def embed_drug(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray().T


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
