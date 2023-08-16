Installation
------------

The easiest way to install **pyethnicity** is through **pip**. Simply run

.. code-block:: bash

    pip install pyethnicity

Note that **pyethnicity** depends on several packages

1. `onnxruntime-gpu`_: For fast and flexible inference
2. `pandas`_: For the final output DataFrames
3. `polars`_: For fast and memory efficient cleaning routines
4. `pyarrow`_: For parquet files
5. `pycutils`_: A lightweight, stdlib-only collection of useful functions
6. `tqdm`_: A lightweight progress bar

.. _onnxruntime-gpu: https://onnxruntime.ai/
.. _pandas: https://pandas.pydata.org/
.. _polars: https://www.pola.rs/
.. _pyarrow: https://arrow.apache.org/docs/python/index.html
.. _pycutils: https://github.com/CangyuanLi/cutils
.. _tqdm: https://github.com/tqdm/tqdm