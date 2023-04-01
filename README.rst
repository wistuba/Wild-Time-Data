Wild-Time-Data: Easy access to the Wild-Time data
*************************************************

This repository provides a simpler interface to access the
`Wild-Time datasets <https://github.com/huaxiuyao/Wild-Time>`_ in PyTorch.
In contrast to the original repository, this repository contains only code relevant for data loading
and has fewer dependencies.

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
* ``data_dir``: Location where to store the data. By default it will be downloaded to ``~/wild-time-data/``.

Other Useful Functions
======================

Several other functions can be imported from ``wild_time_data``.

.. code-block:: python

  from wild_time_data import available_time_steps, input_dim, list_datasets, num_outputs

* ``available_time_steps``: Given the dataset name, a sorted list of available time steps is returned. Example: ``available_time_steps("huffpost")`` returns ``[2012, 2013, 2014, 2015, 2016, 2017, 2018]``.
* ``input_dim``: Given the dataset name, the input dimensionality is returned. For image datasets the shape of the image is returned. For text datasets the maximum number of words separated by spaces is returned. Example: ``input_dim("yearbook")`` returns ``(3, 32, 32)``.
* ``list_datasets``: Returns the list of all available datasets. Example: ``list_datasets()`` returns ``["arxiv", "drug", "fmow", "huffpost", "yearbook"]``.
* ``num_outputs``: Given the dataset name, either the number of classes is returned or it returns 1. In cases where 1 is returned, this indicates that this is a regression dataset. Example: ``num_outputs("arxiv")`` returns ``172``.

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
