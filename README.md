[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE)
# EFTPipe
EFTofLSS analysis pipeline

Developed by [Ruiyang Zhao](mailto:zhaoruiyang19@mails.ucas.edu.cn) and [Xiaoyong Mu](mailto:mouxiaoyong15@mails.ucas.edu.cn)

# Dependencies
- `numpy`, `scipy` and `pandas`
- `cobaya` for likelihood and logging
- `typing_extensions` for type hints, optional
- `camb` or `classy` boltzamnn code, optional
- `numba` for better performance, optional
- `matryoshka` for [emulator](https://github.com/JDonaldM/Matryoshka), optional

`cobaya` will be an optional dependency in the future
# Installation
## Install all relevant packages using Anaconda
The easiest way of setting up the environment and reproducing analysis results in the paper would be using Anaconda
```bash
git clone git@github.com:zhaoruiyang98/eftpipe.git
cd eftpipe
conda env create -f environment.yml
# or `conda env create -f environment-dev.yml` for development
# or `conda env create -f environment-emu-dev.yml` for emulator
```
MPI support could be installed by running
```shell
conda install -c conda-forge "mpi4py>=3"
# or `conda install -c intel "mpi4py>=3"` if you are using intel compiler
# or `pip install "mpi4py>=3" --upgrade --no-binary :all:` if you want to build from source
```

Install classy (for macOS users, please make sure your `gcc` points to gcc instead of clang when compiling classy, or manually edit the `Makefile` to select the proper compiler)
```shell
git clone --depth 1 --branch v3.2.0 https://github.com/lesgourg/class_public
cd class_public/
make
```
## Minimum installation
If `numpy, scipy, pandas` and `cobaya` has been installed, run the following codes to install eftpipe locally
```shell
git clone https://github.com/zhaoruiyang98/eftpipe.git
cd eftpipe
pip install -e .
```
# User guide
Please have a look at notebooks in the `notebook` folder.
# Compare to upstream PyBird
Since EFTPipe is developed based on PyBird, it is quite important to keep the modified codes consistent with upstream PyBird repository.

For demonstration, we write a simple random comparison suite in [this](https://github.com/zhaoruiyang98/eftpipe/blob/main/tests/compare/test_theory.py) file. To run it yourself, you should install test framework mentioned above and install the [forked version of PyBird](https://github.com/zhaoruiyang98/pybird/tree/compare) (`compare` branch), where we made some minor changes:
1. make pybird_dev as a package named `pybird`
2. fix overflow and file load error when computing window matrix
3. double the default number of $\mu$ bins when applying AP effect

Then, at the root directory, run `pytest tests/compare/test_theory.py --diffbird 5`, you will see 5 comparison results (4 are random). Typically the relative difference appears like the following figure:

![compare](https://github.com/zhaoruiyang98/eftpipe/blob/main/figures/compare.png)

Generally they are in very good agreement and the appeared residual has two causes:
1. different integral nodes of $k$
2. and the weird peak comes from the inaccurate comparison of floating numbers when applying the mask to window matrix

If you set totally the same integral nodes, the relative difference will be completely negligible.
# Acknowledgements
Thanks to [Pierre Zhang](mailto:pierrexyz@protonmail.com) and [Guido D'Amico](mailto:damico.guido@gmail.com) for developing [PyBird](https://github.com/pierrexyz/pybird). EFTPipe is motivated by and heavily depends on PyBird. For the purpose of extension, we include it as subpackage. Please have a look at [README](https://github.com/zhaoruiyang98/eftpipe/blob/main/eftpipe/pybird/README.md) for detailed information and citation.
