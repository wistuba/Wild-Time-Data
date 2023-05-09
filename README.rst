Wild-Time-Data: Easy access to the Wild-Time data
*************************************************

This repository provides a simpler interface to access the
`Wild-Time datasets <https://github.com/huaxiuyao/Wild-Time>`_ in PyTorch.
In contrast to the original repository, this repository contains only code relevant for data loading
and has fewer dependencies.
Furthermore, we no longer use pickle to store the data but store it as HDF5 which has two advantages.
First, this is more space-efficient, e.g., for the yearbook dataset the data file size changed from 1.7GB to only 37MB.
Second, it addresses a security concern people may have when using unknown pickled Python objects.

.. image:: yearbook.png
   :align: center


Installation
============

Wild-Time-Data is available via PyPI.

.. code-block:: bash

  pip install wild-time-data

Usage
=====
The following code will return a PyTorch dataset for the training partition of the arXiv dataset in 2023.
The data will be downloaded to ``wild-time-data/`` unless it was downloaded into this folder before.

.. code-block:: python

  from wild_time_data import load_dataset

  load_dataset(dataset_name="arxiv", time_step=2023, split="train", data_dir="wild-time-data")

In the following we provide details about the available argument options.

* ``dataset_name``: The options are arxiv, drug, fmow, huffpost, and yearbook. This list can be accessed via

    .. code-block:: python

      from wild_time_data import list_datasets

      list_datasets()

* ``time_step``: Most datasets are grouped by year, this argument will allow you to access the data from different time intervals. The range differs from dataset to dataset. Use following command to get a list of available time steps:

    .. code-block:: python

      from wild_time_data import available_time_steps

      available_time_steps("arxiv")

* ``split``: Selects the partition. Can either be ``train`` or ``test``.
* ``transform``: Defines custom transformations on the predictors of a data point. Can be used to normalize, augment or tokenize data. By default, no transformation is applied for text datasets, image are converted to Tensors via ``transforms.ToTensor()``, and the data for ``Drug`` is one-hot encoded. Additionally, ``FMoW`` images are normalized. The default transformation can be accessed via

    .. code-block:: python

      from wild_time_data import default_transform

      default_transform("huffpost")

* ``target_transform``: Same as ``transform`` but for labels. By default, no transformation is applied.
* ``in_memory``: If set to ``True``, all data is loaded in memory. By default, data is loaded from disk which might be slower but requires less memory. For all datasets but ``FMoW`` ``in_memory=True`` should work on most hardware.
* ``data_dir``: Location where to store the data. By default it will be downloaded to ``~/wild-time-data/``.

Other Useful Functions
======================

Several other functions can be imported from ``wild_time_data``.

.. code-block:: python

  from wild_time_data import available_time_steps, input_dim, list_datasets, num_outputs

* ``available_time_steps``: Given the dataset name, a sorted list of available time steps is returned. Example: ``available_time_steps("huffpost")`` returns ``[2012, 2013, 2014, 2015, 2016, 2017, 2018]``.
* ``default_transform``: Given the dataset name, the transformation which is applied to the predictors unless a custom transformation is passed. Example: ``default_transform("yearbook")`` returns ``transfroms.ToTensor()``. If the return value is ``None``, no transformation is applied. In order to override a default transformation, pass ``transform=lambda x: x`` to ``load_dataset``.
* ``input_dim``: Given the dataset name, the input dimensionality is returned. For image datasets the shape of the image is returned. For text datasets the maximum number of words separated by spaces is returned. Example: ``input_dim("yearbook")`` returns ``(1, 32, 32)``.
* ``list_datasets``: Returns the list of all available datasets. Example: ``list_datasets()`` returns ``["arxiv", "drug", "fmow", "huffpost", "yearbook"]``.
* ``num_outputs``: Given the dataset name, either the number of classes is returned or it returns 1. In cases where 1 is returned, this indicates that this is a regression dataset. Example: ``num_outputs("arxiv")`` returns ``172``.


FMoW Dataset
============

If you want to use the FMoW dataset, please follow `the instructions to prepare it <https://github.com/wistuba/Wild-Time-Data/tree/main/converter>`__.


Licenses
========
All additional code for Wild-Time-Data is available under the Apache 2.0 license.
The license for each Wild-Time dataset is listed below:

* arXiv: CC0: Public Domain
* Drug-BA: MIT License
* FMoW: `The Functional Map of the World Challenge Public License <https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE>`_
* Huffpost: CC0: Public Domain
* Yearbook: MIT License

Furthermore, this repository is loosely based on the `Wild-Time repository <https://github.com/huaxiuyao/Wild-Time>`_
which is licensed under the MIT License.
