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
    return pd.read_csv(path, **kwargs)  # type: ignore


def read_pkl(
    path: str | os.PathLike, logger: logging.Logger | None = None, **kwargs
) -> pd.DataFrame:
    try:
        df = read_commented_csv(path, **kwargs)
    except NoHeaderError:
        if logger and is_main_process():
            logger.warning("No header found in %s, inferred instead", path)
        names = ["k"] + [f"P{2 * i}" for i in range(np.loadtxt(path).shape[1] - 1)]
        df = read_commented_csv(path, names=names, **kwargs)
    df.set_index(df.columns[0], drop=False, inplace=True)
    return df


def find_reader_else_default(
    name: str | None, default, logger: logging.Logger | None = None, **kwargs
):
    name = name or "default"
    if name == "default":
        return default
    import importlib
    import inspect

    module_name, callable_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, callable_name)

    def reader(path):
        if p := inspect.signature(fn).parameters.get("logger"):
            if p.kind not in [p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD]:
                return fn(path, logger=logger, **kwargs)
        return fn(path, **kwargs)

    return reader


def elephant_reader(root, logger: logging.Logger | None = None, z=1.0):
    from .log import mpi_info
    from pathlib import Path

    logger = logger or logging.getLogger(f"eftpipe.reader.{elephant_reader.__name__}")
    root = Path(root) / "halo/LCDM"
    z = float(z)
    z_snap_mapping = {1.65: 2, 1.4: 3, 1.2: 4, 1.0: 5}
    if (isnap := z_snap_mapping.get(z)) is None:
        raise ValueError(f"z={z} not in {list(z_snap_mapping.keys())}")
    # average over 10 boxes
    it = (
        root / f"Box{i}/zspace_halo/zspace_HaloPk_M12p7_M13p2_snap_{isnap}.txt"
        for i in range(10)
    )
    data = [np.loadtxt(file_name) for file_name in it]
    data = np.mean(data, axis=0)
    k, P0, P2, P4, *_ = data.T
    df = pd.DataFrame({"k": k, "P0": P0, "P2": P2, "P4": P4})
    df.set_index("k", drop=False, inplace=True)
    mpi_info(logger, "Read the average of 10 elephant halo boxes at z=%.3f", z)
    return df


def elephant_cov_reader(root, logger: logging.Logger | None = None, z=1.0):
    from .log import mpi_info
    from pathlib import Path

    logger = logger or logging.getLogger(f"eftpipe.reader.{elephant_reader.__name__}")
    root = Path(root) / "halo/LCDM/AllBoxes/zspace_halo_gcov"
    z = float(z)
    z_snap_mapping = {1.65: 2, 1.4: 3, 1.2: 4, 1.0: 5}
    if (isnap := z_snap_mapping.get(z)) is None:
        raise ValueError(f"z={z} not in {list(z_snap_mapping.keys())}")
    file_name = root / f"zspace_HaloGCov_M12p7_M13p2_snap_{isnap}.txt"
    _, C00, C02, C04, C22, C24, C44 = (
        np.diag(x) for x in np.loadtxt(file_name, unpack=True)
    )
    # fmt: off
    cov = np.block(
        [[C00, C02, C04],
         [C02, C22, C24],
         [C04, C24, C44]]
    )
    # fmt: on
    mpi_info(logger, "Read the elephant halo covariance at z=%.3f", z)
    return cov


def bestfit_reader(yaml_file, logger: logging.Logger | None = None, *, tracer):
    from .analysis import BestfitModel, Multipole

    model = BestfitModel(yaml_file, remove_window=True)
    k = model.multipoles[tracer].k
    fn = model.Plk_interpolator(tracer)
    P0, P2, P4 = fn([0, 2, 4], k)
    return Multipole.init(k=k, P0=P0, P2=P2, P4=P4).data


def bestfit_cov_reader(yaml_file, logger: logging.Logger | None = None, *, tracers):
    from .analysis import BestfitModel, Multipole
    from .covariance import Multipole as Mult
    from .covariance import GaussianCovariance

    model = BestfitModel(yaml_file, remove_window=True)
    truncate = False
    if len(tracers) == 2:
        if "NGC" in tracers[0]:
            tracers.append("X_NGC")
        elif "SGC" in tracers[0]:
            tracers.append("X_SGC")
        else:
            raise NotImplementedError
        truncate = True
    ms = {}
    for tracer in tracers:
        if "LRG" in tracer:
            key = "Paa"
        elif "ELG" in tracer:
            key = "Pbb"
        elif "X" in tracer:
            key = "Pab"
        else:
            raise NotImplementedError
        k = model.multipoles[tracer].k
        fn = model.Plk_interpolator(tracer)
        P0, P2, P4 = fn([0, 2, 4], k)
        if tracer == "LRG_NGC":
            Pshot = 13261.982517118122
        elif tracer == "LRG_SGC":
            Pshot = 12508.022132075843
        elif tracer == "ELG_NGC":
            Pshot = 5297.853491193601
        elif tracer == "ELG_SGC":
            Pshot = 5146.4906395934795
        elif tracer == "X_NGC" or tracer == "X_SGC":
            Pshot = 0.0
        else:
            raise NotImplementedError
        ms[key] = Mult(P0=P0 + Pshot, P2=P2, P4=P4)
    gcov = GaussianCovariance(
        kedges=np.linspace(0, 0.3, 30 + 1), volume=(0.43 + 0.41) * 20
    )
    if len(ms) == 1:
        # single tracer
        return gcov([0, 2, 4], [0, 2, 4], "aa->aaaa", ms.popitem()[-1])
    # multi-tracer
    cov = gcov.fullcov(ms["Paa"], ms["Pbb"], ms["Pab"], ells=[0, 2, 4])
    if truncate:
        nk = cov.shape[0] // 3
        cov = cov[:-nk, :-nk]
    return cov
