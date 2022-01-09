import sys
from setuptools import setup, find_packages
from pathlib import Path


def read_long_description():
    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        text = f.read()
    return text


if sys.version_info >= (3, 8):
    extra_install = []
else:
    extra_install = ['typing_extensions>=3.10']

install_requires = [
    "numpy>=1.20",
    "scipy>=1.6",
    "cobaya>=3.1.1",
    "camb>=1.3.2",
]
install_requires += extra_install

exclude_dir = ['docs', 'tests', 'notes', 'cobaya']
packages = find_packages(exclude=exclude_dir)

test_requires = [
    'pytest>=6.0', 'pytest-cov',
    'pytest-regressions>=2.3.0', 'pytest-datadir',
]

gui_requires = [
    'matplotlib', 'getdist'
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Typing :: Typed",
]

setup(
    name='eftpipe',
    version='0.0.1',
    description='EFTofLSS analysis pipeline for cross power spectrum',
    url='https://github.com/zhaoruiyang98/eftpipe',
    project_urls={
        'Source': 'https://github.com/zhaoruiyang98/eftpipe',
        'Tracker': 'https://github.com/zhaoruiyang98/eftpipe/issues',
        'Licensing': 'https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE'
    },
    author='Ruiyang Zhao and Xiaoyong Mu',
    author_email='zhaoruiyang19@mails.ucas.edu.cn',
    license='MIT',
    python_requires='>=3.7',
    keywords='cosmology perturbation EFT',
    packages=packages,
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
        'gui': gui_requires,
        'dev': test_requires + gui_requires + ['pyyaml>=5.1']
    },
    package_data={
        'eftpipe': ['*.yaml', '*.bibtex', 'py.typed']
    },
    classifiers=classifiers,
    zip_safe=False,
)
