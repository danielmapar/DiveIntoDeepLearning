# Dive Into Deep Learning 

## Installation

* For more details visit: 
    * http://numpy.d2l.ai/chapter_install/install.html

### Steps

* Install [Miniconda](https://conda.io/en/latest/miniconda.html)
    * ```bash
        # For Mac users (the file name is subject to changes)
        sudo sh Miniconda3-latest-MacOSX-x86_64.sh

        # For Linux users (the file name is subject to changes)
        sudo sh Miniconda3-latest-Linux-x86_64.sh

        # For Mac user
        source ~/.bash_profile

        # For Linux user
        source ~/.bashrc

        # Create an env
        conda create --name d2l
        ```
    
    * You can activate/deactivate an env by doing:
        * `conda activate d2l`
        * `conda deactivate`

    * Download book code:
        * For MacOSX only: `brew install wget`
        * ```bash
            sudo apt-get install unzip
            mkdir d2l-en && cd d2l-en
            wget http://numpy.d2l.ai/d2l-en.zip
            unzip d2l-en.zip && rm d2l-en.zip
            ```

    * Within the “d2l” environment, activate it and install pip. Enter y for the following inquiries.
        * ```bash
            conda activate d2l
            conda install pip
            ```

    * Finally, install “d2l” package within the environment “d2l” that we created.
        * `pip install git+https://github.com/d2l-ai/d2l-en@numpy2`

    * At times, to avoid unnecessary repetition, we encapsulate the frequently-imported and referred-to functions, classes, etc. in this book in the d2l package. For any block block such as a function, a class, or multiple imports to be saved in the package, we will mark it with *# Save to the d2l package*. For example, these are the packages and modules will be used by the d2l package.
        * ```python
            # Save to the d2l package
            from IPython import display
            import collections
            import os
            import sys
            import math
            from matplotlib import pyplot as plt
            from mxnet import np, npx, autograd, gluon, init, context, image
            from mxnet.gluon import nn, rnn
            import random
            import re
            import time
            import tarfile
            import zipfile
            ```
    
* Installing MXNet

    * Before installing mxnet, please first check if you are able to access GPUs. If so, please go to [GPU Support](http://numpy.d2l.ai/chapter_install/install.html#sec-gpu) for instructions to install a GPU-supported mxnet. Otherwise, you can install the CPU version, which is still good enough for the first few chapters.
    
    * ```bash
        # For Linux users
        pip install https://apache-mxnet.s3-accelerate.amazonaws.com/dist/python/numpy/latest/mxnet-1.5.0-py2.py3-none-manylinux1_x86_64.whl

        # For macOS with Python 2.7 users
        pip install https://apache-mxnet.s3-us-west-2.amazonaws.com/dist/python/numpy/latest/mxnet-1.5.0-cp27-cp27m-macosx_10_11_x86_64.whl

        # For macOS with Python 3.6 users
        pip install https://apache-mxnet.s3-us-west-2.amazonaws.com/dist/python/numpy/latest/mxnet-1.5.0-cp36-cp36m-macosx_10_11_x86_64.whl

        # For macOS with Python 3.7 users
        pip install https://apache-mxnet.s3-us-west-2.amazonaws.com/dist/python/numpy/latest/mxnet-1.5.0-cp37-cp37m-macosx_10_11_x86_64.whl
        ```

    * Once both packages are installed, we now open the Jupyter notebook by: `jupyter notebook`

