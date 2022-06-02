import re
from setuptools import setup, find_packages
from pathlib import Path


def find_version():
    contents = open(Path(__file__).parent / "eftpipe" / "__init__.py").read()
    pattern = re.compile(r"__version__ = \"(.*?)\"")
    match = pattern.search(contents)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_long_description():
    with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
        text = f.read()
    return text


install_requires = [
    "numpy>=1.21",
    "scipy>=1.6",
    "cobaya>=3.2.1",
    "typing_extensions>=4.1,<5",
]

exclude_dir = ["docs", "tests", "notes", "cobaya"]
packages = find_packages(exclude=exclude_dir)

test_requires = [
    "pytest>=6.0",
    "pytest-cov",
    "pytest-regressions>=2.3.0",
    "pytest-datadir",
]

gui_requires = ["matplotlib", "getdist"]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Typing :: Typed",
]

setup(
    name="eftpipe",
    version=find_version(),
    description="EFTofLSS analysis pipeline",
    long_description=read_long_description(),
    url="https://github.com/zhaoruiyang98/eftpipe",
    project_urls={
        "Source": "https://github.com/zhaoruiyang98/eftpipe",
        "Tracker": "https://github.com/zhaoruiyang98/eftpipe/issues",
        "Licensing": "https://github.com/zhaoruiyang98/eftpipe/blob/main/LICENSE",
    },
    author="Ruiyang Zhao and Xiaoyong Mu",
    author_email="zhaoruiyang19@mails.ucas.edu.cn",
    license="MIT",
    python_requires=">=3.8",
    keywords="cosmology perturbation EFT",
    packages=packages,
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "gui": gui_requires,
        "dev": test_requires + gui_requires + ["pyyaml>=5.1", "line_profiler"],
    },
    package_data={"eftpipe": ["*.yaml", "*.bibtex", "py.typed"]},
    classifiers=classifiers,
    zip_safe=False,
)
