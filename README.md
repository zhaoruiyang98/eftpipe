[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE)
# EFTPipe
EFTofLSS analysis pipeline

Developed by [Ruiyang Zhao](mailto:zhaoruiyang19@mails.ucas.edu.cn) and [Xiaoyong Mu](mailto:mouxiaoyong15@mails.ucas.edu.cn)

# Dependencies
- `numpy` and `scipy` for numerical computing
- `cobaya` for likelihood and logging
- `typing_extensions` for better type hints
- `camb` or `classy` boltzamnn code (not necessary if you only want to run `eftpipe.pybird`)
# Installation
## Installing all packages for analysis using Anaconda
EFTPipe provides the theory and likelihood component for bayesian analysis of large-scale structure data. Typically people need extra packages for plotting and monte-carlo sampling, such as [getdist](https://getdist.readthedocs.io/en/latest/) and [cobaya](https://cobaya.readthedocs.io/en/latest/).

You can use the following codes to install all the packages you may need when doing data analysis.

```bash
# create a new environment
conda create -n eftpipe python=3.8
conda activate eftpipe
# install getdist first
conda install -c conda-forge "numpy>=1.20" "scipy>=1.6" pandas portalocker matplotlib PySide2 tqdm
pip install getdist
# test if getdist is working
python -m unittest getdist.tests.getdist_test
# test if getdist-gui is working
# getdist-gui
# install cobaya
python -m pip install "cobaya>=3.2.1" --upgrade
# install cosmological codes and data
cobaya-install cosmo -p /path/to/packages --upgrade
# install camb equipped with cobaya (ensure latest)
cd /path/to/packages/code/CAMB
pip install .
```

MPI installation guide can be found at cobaya's [website](https://cobaya.readthedocs.io/en/latest/installation.html)

If you want to install test framework for developments, please run the following commands:

```shell
conda install "pytest>=6.0"
conda install -c conda-forge pytest-cov "pytest-regressions>=2.3.0" pytest-datadir
```

Finally, install eftpipe
```shell
# install eftpipe locally
cd /path/to/working/directory
git clone https://github.com/zhaoruiyang98/eftpipe.git
cd eftpipe
pip install -e .
```
## Minimum installation
Please have a look at [setup.py](https://github.com/zhaoruiyang98/eftpipe/blob/main/setup.py) file and make sure you have installed all dependencies and satisfied all version requirements. Then run

```shell
# install eftpipe locally
cd /path/to/working/directory
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