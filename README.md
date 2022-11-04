[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE)
# EFTPipe
EFTofLSS analysis pipeline

Developed by [Ruiyang Zhao](mailto:zhaoruiyang19@mails.ucas.edu.cn) and [Xiaoyong Mu](mailto:mouxiaoyong15@mails.ucas.edu.cn)

# Dependencies
- `numpy` and `scipy` for numerical computing
- `cobaya` for likelihood and logging
- `typing_extensions` for better type hints
- `camb` or `classy` boltzamnn code (not necessary if you only want to run `eftpipe.pybird`)
- `numba` for better performance, optional
- `pandas` for large csv file loading, optional
# Installation
## Installing all packages for analysis using Anaconda
EFTPipe provides the theory and likelihood component for bayesian analysis of large-scale structure data. Typically people need extra packages for plotting, i.e., [getdist](https://getdist.readthedocs.io/en/latest/).

You can use the following codes to install all the packages you may need when doing data analysis.
```bash
git clone git@github.com:zhaoruiyang98/eftpipe.git
cd eftpipe
conda env create -f environment.yml
# or `conda env create -f environment-dev.yml` for development
```
Creating an environment from an environment.yml file is usually very slow, and you may get stuck at solving environment. Alternatively, you can try [mamba](https://mamba.readthedocs.io/en/latest/index.html):
```bash
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
# or `mamba env create -f environment-dev.yml` for development
```

MPI installation guide can be found at cobaya's [website](https://cobaya.readthedocs.io/en/latest/installation.html), please run
```shell
conda install -c conda-forge "mpi4py>=3"
# or `conda install -c intel "mpi4py>=3"` if you are using intel compiler
# or `pip install "mpi4py>=3" --upgrade --no-binary :all:` if you want to build from source
```

Before running MCMC, you may need install some cosmology codes. Cobaya provides an automatic installer and you can use [that](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html) to install `camb` and `classy`. Otherwise you can run the following code (requires gcc) to install `camb` and `classy` (for macOS user, please make sure your `gcc` points to gcc instead of clang when compiling classy)
```shell
pip install "camb>=1.3.5"
git clone --depth 1 --branch v3.2.0 https://github.com/lesgourg/class_public
conda install -c conda-forge cython
cd class_public/
make
```
## Minimum installation
Please run the following commands
```shell
# install eftpipe locally
git clone https://github.com/zhaoruiyang98/eftpipe.git
cd eftpipe
pip install -e .
# or `pip install -e .[test]` for test
# or `pip install -e .[dev]` for development
```
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
