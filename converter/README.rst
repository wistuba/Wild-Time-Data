Creating fmow.hdf5
******************

The FMoW dataset is significantly larger than the other datasets. For that reason, we do not host it
but provide tooling so you can create it yourself.

Step 1: Dependencies
====================

Please make sure that you have all dependencies required installed. Please note, that you do not
need to have ``wild-time-data`` installed for this step and that this is a different requirements
file than used for ``wild-time-data``.

.. code-block:: bash

  pip install -r requirements.txt

Step 2: Create the HDF5
=======================

Next, create the HDF5 file. You can define the location for temporary and final files via the
``data_dir`` argument.

.. code-block:: bash

  python fmow.py --data-dir output

Step 3: Load with ``wild-time-data``
====================================

Now you can load FMoW as every other dataset with ``wild-time-data``.

.. code-block:: python

  from wild_time_data import load_dataset

  load_dataset(dataset_name="fmow", time_step=0, split="train", data_dir="output")
