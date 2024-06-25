[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE)
# eftpipe
> [!WARNING]  
> The code still needs to be cleaned up, expected to finish before mid-July.
> 
> Future developments will be moved to [desilike](https://github.com/cosmodesi/desilike), and all changes will be directly submitted to the upstream [PyBird](https://github.com/pierrexyz/pybird) repo.

A Python package for doing the multi-tracer EFT analysis (arXiv: [2308.06206](https://arxiv.org/abs/2308.06206)).

If you have any questions, please feel free to contact [Ruiyang Zhao](mailto:zhaoruiyang19@mails.ucas.edu.cn) and [Xiaoyong Mu](mailto:mouxiaoyong15@mails.ucas.edu.cn). Any additional products are also available upon reasonable request.

## Dependencies
- python>=3.10
- `numpy`, `scipy` and `pandas`
- `cobaya` for MCMC sampling
- `typing_extensions` for type hints, optional
- `camb` or `classy` boltzamnn code, optional
- `numba` for better performance, optional
- `matryoshka` for [emulator](https://github.com/JDonaldM/Matryoshka), optional, not well tested yet

## Installation
### Install all packages using Anaconda
The easiest way to set up the environment and reproduce analysis results in the paper would be using Anaconda
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
# or `conda install -c intel "mpi4py>=3"` if you are using the intel compiler
# or `pip install "mpi4py>=3" --upgrade --no-binary :all:` if you want to build from source
```

Install classy
```shell
git clone --depth 1 --branch v3.2.2 https://github.com/lesgourg/class_public
cd class_public/
# For mac user: you have to edit class_public/Makefile line21 by hand to set CC to the path to your gcc compiler (gcc points to clang on mac by default)
make
```
### Minimum installation
If `numpy, scipy, pandas` and `cobaya` have already been installed, you could run the following code to install eftpipe locally
```shell
git clone https://github.com/zhaoruiyang98/eftpipe.git
cd eftpipe
pip install -e .
```

## Acknowledgements
- Pierre Zhang and Guido D'Amico for developing PyBird and making this wonderful code public. eftpipe heavily relies on PyBird. For the purpose of extension, we include it as a subpackage. Please have a look at [README](https://github.com/zhaoruiyang98/eftpipe/blob/main/eftpipe/pybird/README.md) for detailed information and citation.
- Cheng Zhao for providing EZmock catalogues.
- Arnaud de Mattia for developing the [pypower](https://github.com/cosmodesi/pypower) package, which is used for power spectrum estimation in this work.