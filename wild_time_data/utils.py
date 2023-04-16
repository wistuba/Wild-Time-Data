from hashlib import blake2b
from pathlib import Path

import gdown


def checksum_check(file, checksum):
    hashing_function = blake2b()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hashing_function.block_size), b""):
            hashing_function.update(chunk)
    return hashing_function.hexdigest() == checksum


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
