import os

from wild_time_data.datasets import ArXiv, Drug, FMoW, HuffPost, Yearbook

dataset_classes = {
    "arxiv": ArXiv,
    "drug": Drug,
    "fmow": FMoW,
    "huffpost": HuffPost,
    "yearbook": Yearbook,
}


def _check_dataset_exists(dataset_name):
    if dataset_name not in dataset_classes:
        raise ValueError(
            f"Unknown dataset {dataset_name}. Available datasets are {dataset_classes.keys()}"
        )


def available_time_steps(dataset_name):
    """Lists all available time steps."""
    _check_dataset_exists(dataset_name)
    return dataset_classes[dataset_name].time_steps


def input_dim(dataset_name):
    """Returns the input dimensionality of the data.

    Image datasets: image shape
    Text datasets: highest number of words in a sentence
    Protein datasets: list of two tuples
    """
    _check_dataset_exists(dataset_name)
    return dataset_classes[dataset_name].input_dim


def num_outputs(dataset_name):
    """Returns the number of classes for a classification task (num_outputs > 1) and 1 for a regression task."""
    _check_dataset_exists(dataset_name)
    return dataset_classes[dataset_name].num_outputs


def default_transform(dataset_name):
    """Returns the default transformation applied to the predictors."""
    _check_dataset_exists(dataset_name)
    return dataset_classes[dataset_name].default_transform


def list_datasets():
    """Lists all available dataset names."""
    return list(dataset_classes.keys())


def load_dataset(
    dataset_name,
    time_step,
    split,
    transform=None,
    target_transform=None,
    in_memory=False,
    data_dir=os.path.expanduser("~/wild-time-data"),
):
    """Loads the different Wild-Time datasets.

    Args:
        dataset_name: The name of the respective dataset.
        time_step: Indicate the time slice. Available time slices can be accessed via ``available_time_steps()``.
        split: Whether to load train or test split.
        transform: Change default transformations on the predictors of a data point. Can be used to normalize, augment or tokenize data.
        target_transform: Add a transformation to the labels.
        in_memory: If ``True``, the entire data will be loaded into memory.
        data_dir: Location where the data is stored.
    """
    _check_dataset_exists(dataset_name)
    if split not in ["train", "test"]:
        raise ValueError("Only available splits are train and test.")

    return dataset_classes[dataset_name](
        time_step=time_step,
        split=split,
        transform=transform,
        target_transform=target_transform,
        in_memory=in_memory,
        data_dir=data_dir,
    )
