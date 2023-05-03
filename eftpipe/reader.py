from __future__ import annotations
import logging
import os
import numpy as np
import pandas as pd
from .tools import is_main_process


class NoHeaderError(ValueError):
    pass


def read_commented_csv(
    path: str | os.PathLike, names: list | None = None, **pandas_kwargs
) -> pd.DataFrame:
    """numpy compatible csv reader"""
    if not names:
        with open(path) as f:
            header = f.readline()
            if header.startswith("#"):
                names = header[1:].strip().split()
    if not names:
        raise NoHeaderError(f"No commented header found in {path}")
    kwargs = dict(comment="#", delim_whitespace=True, names=names)
    kwargs.update(pandas_kwargs)
    return pd.read_csv(path, **kwargs)


def read_pkl(
    path: str | os.PathLike, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """numpy compatible csv reader"""
    try:
        df = read_commented_csv(path)
    except NoHeaderError:
        if logger and is_main_process():
            logger.warning("No header found in %s, inferred instead", path)
        names = ["k"] + [f"P{2 * i}" for i in range(np.loadtxt(path).shape[1] - 1)]
        df = read_commented_csv(path, names=names)
    df.set_index("k", drop=False, inplace=True)
    return df
