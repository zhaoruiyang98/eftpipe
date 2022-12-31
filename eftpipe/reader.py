from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .tools import is_main_process


def read_pkl(path: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    """numpy compatible csv reader"""
    with open(path) as f:
        header = f.readline()
        if not header.startswith("#"):
            names = []
        else:
            names = header.strip().split()[1:]
    if not names:
        if logger and is_main_process():
            logger.warning("No header found in %s, inferred instead", path)
        tmp = np.loadtxt(path)
        names = ["k"] + [f"P{2 * i}" for i in range(tmp.shape[1] - 1)]
    ret = pd.read_csv(path, comment="#", delim_whitespace=True, names=names)
    ret.set_index("k", drop=False, inplace=True)
    return ret
