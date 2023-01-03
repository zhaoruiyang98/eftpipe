# TODO: extract the common part of EFTLikeDouble and EFTLikeDoubleCross
from __future__ import annotations
import importlib
import itertools
import logging
import re
import numpy as np
import pandas as pd
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Union
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood
from .marginal import Marginalizable
from .reader import read_pkl
from .tools import int_or_list
from .tools import pairwise

FloatBound_T = Union[float, List[float], None]


def find_data_reader(
    name: str, kwargs: dict[str, Any], logger: logging.Logger | None = None
):
    if name == "auto":

        def ret(path):
            return read_pkl(path, logger=logger)

    else:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ret = getattr(module, class_name)(**kwargs)
    return ret


def find_covariance_reader(name: str, kwargs: dict[str, Any]):
    if name == "auto":
        ret = np.loadtxt
    else:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ret = getattr(module, class_name)(**kwargs)
    return ret


def extract_multipole_info(names: Iterable[str]) -> tuple[str, list[int]]:
    pattern = re.compile(r"^([A-Za-z]+)(\d+)$")
    symbols = set()
    ells: list[int] = []
    for x in names:
        if match := pattern.match(x):
            s, e = match.groups()
            symbols.add(s)
            ells.append(int(e))
    if len(symbols) == 0:
        raise ValueError(f"no valid multipole name found")
    elif len(symbols) != 1:
        raise ValueError(f"ambiguous multipole names: {symbols}")
    ells.sort()
    return symbols.pop(), ells


def regularize_float_bound(
    x: FloatBound_T, n: int, default: float | None = None
) -> list[float]:
    if x is None:
        if default is None:
            raise TypeError("empty bound is not allowed if default is not provided")
        return [default] * n
    elif isinstance(x, float):
        return [x] * n
    else:
        if len(x) != n:
            raise ValueError(f"expect len(x) = {n}, obtained {len(x)}")
        return list(x)


def parse_kmask(
    kall,
    ells: list[int],
    kmin: FloatBound_T,
    kmax: FloatBound_T,
) -> dict[int, slice]:
    """generate kmask for each ell

    Parameters
    ----------
    kall : array-like, 1d
        should be sorted in ascending order
    ells : list[int]
        list of ells
    kmin : float | list[float] | None
        minimum k for each ell, if None, do not have lower bound (use -1)
    kmax : float | list[float] | None
        maximum k for each ell, if None, do not have upper bound (use 1e10)

    Returns
    -------
    dict[int, slice]
        keys are sorted in ascending order
    """
    nl = len(ells)
    ret: dict[int, slice] = {}
    try:
        kminlst = regularize_float_bound(kmin, nl, -1)
        kmaxlst = regularize_float_bound(kmax, nl, 1e10)
    except ValueError as ex:
        raise ValueError(f"length of kmin/kmax does not match ells") from ex
    for ell, min_kmask, max_kmask in zip(sorted(ells), kminlst, kmaxlst):
        ileft = np.searchsorted(kall, min_kmask)
        iright = np.searchsorted(kall, max_kmask, side="right")
        ret[ell] = slice(ileft, iright)
    return ret


def slice_union(slices: Iterable[slice]) -> slice:
    start = min(s.start for s in slices)
    stop = max(s.stop for s in slices)
    return slice(start, stop)


def mask_covariance(cov, *args):
    """generate masked covariance matrix

    Parameters
    ----------
    cov : ndarray, 2d
        covariance matrix
    *args : ``ls``, ``ls_tot``, ``kall``, ``kmin``, ``kmax`` in sequence
        ls : list[int]
            selected multipole order
        ls_tot : list[int]
            all multipole order
        kall : array-like, 1d
            total wavenumber
        kmin : float | list[float] | None
            minimum k for each ell, if None, do not have lower bound (use -1)
        kmax : float | list[float] | None
            maximum k for each ell, if None, do not have upper bound (use 1e10)

    Notes
    -----
    ``ls``, ``ls_tot`` and ``kall`` should be sorted in ascending order
    """
    mask1d = np.array([], dtype=bool)
    remaining = list(args)
    while remaining:
        ls, ls_tot, kall, kmin, kmax, *remaining = remaining
        kmask = parse_kmask(kall, ls, kmin, kmax)
        nl, nk = len(ls_tot), len(kall)
        mask = np.zeros((nl, nk), dtype=bool)
        for i, ell in enumerate(ls_tot):
            mask[i, kmask.get(ell, [])] = True
        mask1d = np.hstack((mask1d, mask.flatten()))
    if cov.shape[0] != mask1d.size:
        raise ValueError(
            f"covariance matrix's shape {cov.shape} does not match input data, "
            f"expected {(mask1d.size, mask1d.size)}"
        )
    return cov[np.outer(mask1d, mask1d)].reshape(mask1d.sum(), -1)


def hartlap(Nreal: int, ndata: int) -> float:
    return (Nreal - ndata - 2) / (Nreal - 1)


def flatten(
    ls: list[int], array, mask: dict[int, slice] | None = None, out=None
) -> Any:
    """

    Parameters
    ----------
    ls : list[int]
        selected multipoles, should be even
    array : ndarray, 2d
    mask : dict[int, slice] | None
        mask for each ell, if None, use all
    out : ndarray, 1d, optional
        output array, if None, create a new one
    """
    idx = [ell // 2 for ell in ls]
    if out is None:
        if not mask:
            return array[idx, :].flatten()
        return np.hstack([array[i, mask[ell]] for i, ell in zip(idx, ls)])
    if not mask:
        istart, nlen = 0, array.shape[1]
        for i in idx:
            out[istart : istart + nlen] = array[i, :]
            istart += nlen
        return
    istart = 0
    for i, ell in zip(idx, ls):
        nlen = mask[ell].stop - mask[ell].start
        out[istart : istart + nlen] = array[i, mask[ell]]
        istart += nlen


@dataclass
class MultipoleInfo:
    df: pd.DataFrame = field(repr=False)
    symbol: str
    ls: list[int]
    ls_tot: list[int] = field(repr=False)
    kmin: FloatBound_T
    kmax: FloatBound_T
    kmask: dict[int, slice] = field(repr=False)
    kout: Any = field(repr=False)  # ndarray, 1d
    kout_mask: dict[int, slice] = field(repr=False)
    data_vector: Any = field(repr=False)  # ndarray, 1d

    @classmethod
    def load(
        cls,
        path: str,
        ls: int | list[int],
        kmin: FloatBound_T = None,
        kmax: FloatBound_T = None,
        reader: str = "auto",
        reader_kwargs: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ):
        df = find_data_reader(
            reader,
            reader_kwargs,
            logger=logger,
        )(path)
        symbol, ls_tot = extract_multipole_info(df.columns)
        ls = int_or_list(ls)
        if not_existed_ls := set(ls).difference(ls_tot):
            raise ValueError(f"ls {not_existed_ls} not found in data")
        kmask = parse_kmask(df.index, ls, kmin, kmax)
        data_vector = np.hstack(
            [df[symbol + str(ell)].to_numpy()[kmask[ell]] for ell in ls]
        )
        # kout is the k-grid which the theory code will compute on
        kout = df.index.to_numpy()[slice_union(kmask.values())]
        kout_mask = parse_kmask(kout, ls, kmin, kmax)
        return cls(
            df=df,
            symbol=symbol,
            ls=ls,
            ls_tot=ls_tot,
            kmin=copy(kmin),
            kmax=copy(kmax),
            kmask=kmask,
            kout=kout,
            kout_mask=kout_mask,
            data_vector=data_vector,
        )


class EFTLikeSingle(Likelihood, Marginalizable):
    """EFT likelihood for single tracer"""

    file_base_name = "eftlike_s"

    tracer: str
    data: dict[str, Any]
    cov: dict[str, Any]
    chained: bool
    with_binning: bool
    binning: dict[str, Any]
    marg: dict[str, dict[str, Any]]

    def initialize(self) -> None:
        super().initialize()

        self.minfo = MultipoleInfo.load(**self.data, logger=self.log)
        self.data_vector = self.minfo.data_vector
        self.ndata = self.data_vector.size
        self.binning = self.binning or {}
        self.binning = {"kout": self.minfo.kout, **self.binning}

        self.set_invcov()
        self.log_data_loading_info()

    def set_invcov(self) -> None:
        if not isinstance(self.cov, dict):
            self.cov = {"path": self.cov}
        cov = find_covariance_reader(
            self.cov.get("reader", "auto"), self.cov.get("reader_kwargs", {})
        )(self.cov["path"])
        cov = mask_covariance(
            cov,
            self.minfo.ls,
            self.minfo.ls_tot,
            self.minfo.df.index,
            self.minfo.kmin,
            self.minfo.kmax,
        )
        cov /= self.cov.get("rescale", 1)
        self.invcov = np.linalg.inv(cov)
        self.hartlap: float | None = None
        if (Nreal := self.cov.get("Nreal")) is not None:
            self.hartlap = hartlap(Nreal, self.ndata)
            self.invcov *= self.hartlap

    def set_cache(self) -> None:
        self._PNG_cache = np.zeros(self.data_vector.size)
        if self.marg:
            self._PG_cache = np.zeros((len(self.valid_prior), self.data_vector.size))

    def log_data_loading_info(self) -> None:
        self.mpi_info("data ells=%s", self.minfo.ls)
        for ell in self.minfo.ls:
            kmasked = self.minfo.kout[self.minfo.kout_mask[ell]]
            self.mpi_info(
                "ell=%d, kmin=%.3e, kmax=%.3e, ndata=%d",
                ell,
                kmasked[0],
                kmasked[-1],
                kmasked.size,
            )
        self.mpi_info("data vector size=%d", self.ndata)
        self.mpi_info(
            "Hartlap correction: %s",
            "off" if not self.hartlap else f"{self.hartlap:.3f}",
        )
        self.mpi_info("covariance rescale: %.3e", self.cov.get("rescale", 1))

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)
        # XXX: is it possible to get rid of hardcoded prefix?
        self.prefix = self.provider.model.theory["eftpipe.eftlss." + self.tracer].prefix
        if self.marg:
            self.setup_prior(self.marg)
            self.report_marginalized()
        self.set_cache()

    def get_requirements(self):
        reqs = {}
        if self.with_binning:
            reqs["nonlinear_Plk_grid"] = {
                self.tracer: {
                    "ls": self.minfo.ls,
                    "chained": self.chained,
                    "binned": True,
                    "binning": self.binning,
                }
            }
        else:
            reqs["nonlinear_Plk_interpolator"] = {
                self.tracer: {
                    "ls": self.minfo.ls,
                    "chained": self.chained,
                }
            }
        if self.marg:
            reqs["nonlinear_Plk_gaussian_grid"] = {
                self.tracer: {
                    "ls": self.minfo.ls,
                    "chained": self.chained,
                    "binned": self.with_binning,
                    **({"binning": self.binning} if self.with_binning else {}),
                }
            }
        return reqs

    # override
    def marginalizable_params(self) -> list[str]:
        return [
            self.prefix + name
            for name in ("b3", "cct", "cr1", "cr2", "ce0", "cemono", "cequad")
        ]

    # override
    def PG(self):
        _, kgrid, table = self.provider.get_nonlinear_Plk_gaussian_grid(
            self.tracer, chained=self.chained, binned=self.with_binning
        )
        for i, bG in enumerate(self.valid_prior.keys()):
            plk = table[bG]
            if not self.with_binning:
                interpfn = interp1d(kgrid, kgrid * plk, kind="cubic", axis=-1)
                fn = lambda k: interpfn(k) / k
                plk = fn(self.minfo.kout)
            flatten(self.minfo.ls, plk, self.minfo.kout_mask, out=self._PG_cache[i])
        return self._PG_cache

    # override
    def PNG(self) -> Any:
        if self.with_binning:
            _, _, plk = self.provider.get_nonlinear_Plk_grid(
                self.tracer, chained=self.chained, binned=self.with_binning
            )
        else:
            fn = self.provider.get_nonlinear_Plk_interpolator(
                self.tracer, chained=self.chained
            )
            plk = fn(self.minfo.ls, self.minfo.kout)
        flatten(self.minfo.ls, plk, self.minfo.kout_mask, out=self._PNG_cache)
        return self._PNG_cache

    # override
    def get_data_vector(self):
        return self.data_vector

    # override
    def get_invcov(self):
        return self.invcov

    def calculate(self, state, want_derived=True, **params_values_dict):
        if self.marg:
            state["logp"] = self.marginalized_logp()
        else:
            res = self.data_vector - self.PNG()
            chi2 = res @ self.invcov @ res
            state["logp"] = -0.5 * chi2


class EFTLikeDouble(Likelihood, Marginalizable):
    """EFT likelihood for two tracers"""

    file_base_name = "eftlike_d"

    tracer: list[str]  # NOTE: intended not to use plural to keep interface consistent
    data: dict[str, dict[str, Any]]  # also support list[dict[str, Any]]
    cov: dict[str, Any]
    chained: dict[str, bool]  # also support bool | list[bool]
    with_binning: dict[str, bool]  # also support bool | list[bool]
    binning: dict[str, dict[str, Any]]  # also support list[dict[str, Any]]
    marg: dict[str, dict[str, Any]]

    def initialize(self) -> None:
        super().initialize()
        self.regularize_attributes()
        self.minfodict = {
            t: MultipoleInfo.load(**self.data[t], logger=self.log) for t in self.tracer
        }
        self.data_vector = np.hstack(
            [minfo.data_vector for minfo in self.minfodict.values()]
        )
        self.ndata = self.data_vector.size
        for t, minfo in self.minfodict.items():
            self.binning[t]["kout"] = minfo.kout

        self.set_invcov()
        self.log_data_loading_info()

    def regularize_attributes(self):
        if len(self.tracer) != 2:
            raise ValueError(f"expected two tracers, got {self.tracer}")

        def list_to_dict(x: Any) -> Any:
            if not isinstance(x, (dict, list)):
                x = [x, x]
            if isinstance(x, list):
                return {t: x[i] for i, t in enumerate(self.tracer)}
            return x

        self.data: dict[str, dict[str, Any]] = list_to_dict(self.data)
        self.chained: dict[str, bool] = list_to_dict(self.chained)
        self.with_binning: dict[str, bool] = list_to_dict(self.with_binning)
        self.binning = self.binning or {t: {} for t in self.tracer}
        self.binning: dict[str, dict[str, Any]] = list_to_dict(self.binning)

    def set_invcov(self) -> None:
        if not isinstance(self.cov, dict):
            self.cov = {"path": self.cov}
        cov = find_covariance_reader(
            self.cov.get("reader", "auto"), self.cov.get("reader_kwargs", {})
        )(self.cov["path"])
        args: Any = ()
        for minfo in self.minfodict.values():
            args += (
                minfo.ls,
                minfo.ls_tot,
                minfo.df.index,
                minfo.kmin,
                minfo.kmax,
            )
        cov = mask_covariance(cov, *args)
        cov /= self.cov.get("rescale", 1)
        self.invcov = np.linalg.inv(cov)
        self.hartlap: float | None = None
        if (Nreal := self.cov.get("Nreal")) is not None:
            self.hartlap = hartlap(Nreal, self.ndata)
            self.invcov *= self.hartlap

    def log_data_loading_info(self) -> None:
        for t, minfo in self.minfodict.items():
            self.mpi_info("tracer %s:", t)
            for ell in minfo.ls:
                kmasked = minfo.kout[minfo.kout_mask[ell]]
                self.mpi_info(
                    "ell=%d, kmin=%.3e, kmax=%.3e, ndata=%d",
                    ell,
                    kmasked[0],
                    kmasked[-1],
                    kmasked.size,
                )
        self.mpi_info("data vector size=%d", self.ndata)
        self.mpi_info(
            "Hartlap correction: %s",
            "off" if not self.hartlap else f"{self.hartlap:.3f}",
        )
        self.mpi_info("covariance rescale: %.3e", self.cov.get("rescale", 1))

    def set_cache(self) -> None:
        self._PNG_cache = np.zeros(self.data_vector.size)
        self._istart_iend_cache: list[tuple[int, int]] = [
            (0, data_vector0_size := self.minfodict[self.tracer[0]].data_vector.size),
            (data_vector0_size, self.data_vector.size),
        ]
        if self.marg:
            self._PG_cache = np.zeros((len(self.valid_prior), self.data_vector.size))
            bG_group: list[list[str]] = [[], []]
            prefix0 = self.prefix[0]
            for name in self.valid_prior.keys():
                if name.startswith(prefix0):
                    bG_group[0].append(name)
                else:
                    bG_group[1].append(name)
            self._bG_group_cache = bG_group

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)
        self.prefix: list[str] = [
            self.provider.model.theory["eftpipe.eftlss." + t].prefix
            for t in self.tracer
        ]
        if self.marg:
            self.setup_prior(self.marg)
            self.report_marginalized()
        self.set_cache()

    def get_requirements(self):
        reqs = {
            "nonlinear_Plk_grid": {},
            "nonlinear_Plk_interpolator": {},
            "nonlinear_Plk_gaussian_grid": {},
        }
        for t, minfo, chained, with_binning, binning in zip(
            self.tracer,
            self.minfodict.values(),
            self.chained.values(),
            self.with_binning.values(),
            self.binning.values(),
        ):
            if with_binning:
                reqs["nonlinear_Plk_grid"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                    "binned": True,
                    "binning": binning,
                }
            else:
                reqs["nonlinear_Plk_interpolator"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                }
            # XXX: require gaussian grid for all tracers, which may not be necessary,
            # but I don't know how to fix this, because tracers' prefixes are not known
            if self.marg:
                reqs["nonlinear_Plk_gaussian_grid"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                    "binned": with_binning,
                    **({"binning": binning} if with_binning else {}),
                }
        reqs = {k: v for k, v in reqs.items() if v}
        return reqs

    # override
    def marginalizable_params(self) -> list[str]:
        params = []
        for prefix in self.prefix:
            params += [
                prefix + name
                for name in ("b3", "cct", "cr1", "cr2", "ce0", "cemono", "cequad")
            ]
        return params

    # override
    def PG(self):
        for t, minfo, chained, with_binning, bGlist, shift, (istart, iend) in zip(
            self.tracer,
            self.minfodict.values(),
            self.chained.values(),
            self.with_binning.values(),
            self._bG_group_cache,
            (0, len(self._bG_group_cache[0])),
            self._istart_iend_cache,
        ):
            _, kgrid, table = self.provider.get_nonlinear_Plk_gaussian_grid(
                t,
                chained=chained,
                binned=with_binning,
            )
            for i, bG in enumerate(bGlist):
                plk = table[bG]
                if not with_binning:
                    interpfn = interp1d(kgrid, kgrid * plk, kind="cubic", axis=-1)
                    fn = lambda k: interpfn(k) / k
                    plk = fn(minfo.kout)
                i += shift
                flatten(
                    minfo.ls, plk, minfo.kout_mask, out=self._PG_cache[i, istart:iend]
                )
        return self._PG_cache

    # override
    def PNG(self) -> Any:
        for t, minfo, with_binning, chained, (istart, iend) in zip(
            self.tracer,
            self.minfodict.values(),
            self.with_binning.values(),
            self.chained.values(),
            self._istart_iend_cache,
        ):
            if with_binning:
                _, _, plk = self.provider.get_nonlinear_Plk_grid(
                    t, chained=chained, binned=True
                )
            else:
                fn = self.provider.get_nonlinear_Plk_interpolator(t, chained=chained)
                plk = fn(minfo.ls, minfo.kout)
            flatten(minfo.ls, plk, minfo.kout_mask, out=self._PNG_cache[istart:iend])
        return self._PNG_cache

    # override
    def get_data_vector(self):
        return self.data_vector

    # override
    def get_invcov(self):
        return self.invcov

    def calculate(self, state, want_derived=True, **params_values_dict):
        if self.marg:
            state["logp"] = self.marginalized_logp()
        else:
            res = self.data_vector - self.PNG()
            chi2 = res @ self.invcov @ res
            state["logp"] = -0.5 * chi2


class EFTLikeDoubleCross(Likelihood, Marginalizable):
    """EFT likelihood for two tracers and cross"""

    file_base_name = "eftlike_dx"

    tracer: list[str]  # NOTE: intended not to use plural to keep interface consistent
    data: dict[str, dict[str, Any]]  # also support list[dict[str, Any]]
    cov: dict[str, Any]
    chained: dict[str, bool]  # also support bool | list[bool]
    with_binning: dict[str, bool]  # also support bool | list[bool]
    binning: dict[str, dict[str, Any]]  # also support list[dict[str, Any]]
    marg: dict[str, dict[str, Any]]

    def initialize(self) -> None:
        super().initialize()
        self.regularize_attributes()
        self.minfodict = {
            t: MultipoleInfo.load(**self.data[t], logger=self.log) for t in self.tracer
        }
        self.data_vector = np.hstack(
            [minfo.data_vector for minfo in self.minfodict.values()]
        )
        self.ndata = self.data_vector.size
        for t, minfo in self.minfodict.items():
            self.binning[t]["kout"] = minfo.kout

        self.set_invcov()
        self.log_data_loading_info()

    def regularize_attributes(self):
        if len(self.tracer) != 3:
            raise ValueError(f"expected two tracers and one cross, got {self.tracer}")

        def list_to_dict(x: Any) -> Any:
            if not isinstance(x, (dict, list)):
                x = [x, x, x]
            if isinstance(x, list):
                return {t: x[i] for i, t in enumerate(self.tracer)}
            return x

        self.data: dict[str, dict[str, Any]] = list_to_dict(self.data)
        self.chained: dict[str, bool] = list_to_dict(self.chained)
        self.with_binning: dict[str, bool] = list_to_dict(self.with_binning)
        self.binning = self.binning or {t: {} for t in self.tracer}
        self.binning: dict[str, dict[str, Any]] = list_to_dict(self.binning)

    def set_invcov(self) -> None:
        if not isinstance(self.cov, dict):
            self.cov = {"path": self.cov}
        cov = find_covariance_reader(
            self.cov.get("reader", "auto"), self.cov.get("reader_kwargs", {})
        )(self.cov["path"])
        args: Any = ()
        for minfo in self.minfodict.values():
            args += (
                minfo.ls,
                minfo.ls_tot,
                minfo.df.index,
                minfo.kmin,
                minfo.kmax,
            )
        cov = mask_covariance(cov, *args)
        cov /= self.cov.get("rescale", 1)
        self.invcov = np.linalg.inv(cov)
        self.hartlap: float | None = None
        if (Nreal := self.cov.get("Nreal")) is not None:
            self.hartlap = hartlap(Nreal, self.ndata)
            self.invcov *= self.hartlap

    def log_data_loading_info(self) -> None:
        for t, minfo in self.minfodict.items():
            self.mpi_info("tracer %s:", t)
            for ell in minfo.ls:
                kmasked = minfo.kout[minfo.kout_mask[ell]]
                self.mpi_info(
                    "ell=%d, kmin=%.3e, kmax=%.3e, ndata=%d",
                    ell,
                    kmasked[0],
                    kmasked[-1],
                    kmasked.size,
                )
        self.mpi_info("data vector size=%d", self.ndata)
        self.mpi_info(
            "Hartlap correction: %s",
            "off" if not self.hartlap else f"{self.hartlap:.3f}",
        )
        self.mpi_info("covariance rescale: %.3e", self.cov.get("rescale", 1))

    def set_cache(self) -> None:
        self._PNG_cache = np.zeros(self.data_vector.size)
        sizelist: list[int] = [
            minfo.data_vector.size for minfo in self.minfodict.values()
        ]
        self._istart_iend_cache: list[tuple[int, int]] = list(
            pairwise(itertools.accumulate(sizelist, initial=0))
        )
        if self.marg:
            self._PG_cache = np.zeros((len(self.valid_prior), self.data_vector.size))
            bG_group: list[list[str]] = [[], [], []]
            prefix0, prefix1 = self.prefix[0], self.prefix[1]
            for name in self.valid_prior.keys():
                if name.startswith(prefix0):
                    bG_group[0].append(name)
                elif name.startswith(prefix1):
                    bG_group[1].append(name)
                else:
                    bG_group[2].append(name)
            self._bG_group_cache = bG_group

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)
        self.prefix: list[str] = [
            self.provider.model.theory["eftpipe.eftlss." + t].prefix
            for t in self.tracer
        ]
        if self.marg:
            self.setup_prior(self.marg)
            self.report_marginalized()
        self.set_cache()

    def get_requirements(self):
        reqs = {
            "nonlinear_Plk_grid": {},
            "nonlinear_Plk_interpolator": {},
            "nonlinear_Plk_gaussian_grid": {},
        }
        for t, minfo, chained, with_binning, binning in zip(
            self.tracer,
            self.minfodict.values(),
            self.chained.values(),
            self.with_binning.values(),
            self.binning.values(),
        ):
            if with_binning:
                reqs["nonlinear_Plk_grid"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                    "binned": True,
                    "binning": binning,
                }
            else:
                reqs["nonlinear_Plk_interpolator"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                }
            # XXX: require gaussian grid for all tracers, which may not be necessary,
            # but I don't know how to fix this, because tracers' prefixes are not known
            if self.marg:
                reqs["nonlinear_Plk_gaussian_grid"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                    "binned": with_binning,
                    **({"binning": binning} if with_binning else {}),
                }
        reqs = {k: v for k, v in reqs.items() if v}
        return reqs

    # override
    def marginalizable_params(self) -> list[str]:
        params = []
        for i, prefix in enumerate(self.prefix):
            if i != 2:
                params += [
                    prefix + name
                    for name in ("b3", "cct", "cr1", "cr2", "ce0", "cemono", "cequad")
                ]
            else:
                params += [prefix + name for name in ("ce0", "cemono", "cequad")]
        return params

    # override
    def PG(self):
        for t, minfo, chained, with_binning, bGlist, shift, (istart, iend) in zip(
            self.tracer,
            self.minfodict.values(),
            self.chained.values(),
            self.with_binning.values(),
            self._bG_group_cache,
            (
                0,
                len(self._bG_group_cache[0]),
                len(self._bG_group_cache[0]) + len(self._bG_group_cache[1]),
            ),
            self._istart_iend_cache,
        ):
            _, kgrid, table = self.provider.get_nonlinear_Plk_gaussian_grid(
                t,
                chained=chained,
                binned=with_binning,
            )
            for i, bG in enumerate(bGlist):
                plk = table[bG]
                if not with_binning:
                    interpfn = interp1d(kgrid, kgrid * plk, kind="cubic", axis=-1)
                    fn = lambda k: interpfn(k) / k
                    plk = fn(minfo.kout)
                i += shift
                flatten(
                    minfo.ls, plk, minfo.kout_mask, out=self._PG_cache[i, istart:iend]
                )
        return self._PG_cache

    # override
    def PNG(self) -> Any:
        for t, minfo, with_binning, chained, (istart, iend) in zip(
            self.tracer,
            self.minfodict.values(),
            self.with_binning.values(),
            self.chained.values(),
            self._istart_iend_cache,
        ):
            if with_binning:
                _, _, plk = self.provider.get_nonlinear_Plk_grid(
                    t, chained=chained, binned=True
                )
            else:
                fn = self.provider.get_nonlinear_Plk_interpolator(t, chained=chained)
                plk = fn(minfo.ls, minfo.kout)
            flatten(minfo.ls, plk, minfo.kout_mask, out=self._PNG_cache[istart:iend])
        return self._PNG_cache

    # override
    def get_data_vector(self):
        return self.data_vector

    # override
    def get_invcov(self):
        return self.invcov

    def calculate(self, state, want_derived=True, **params_values_dict):
        if self.marg:
            state["logp"] = self.marginalized_logp()
        else:
            res = self.data_vector - self.PNG()
            chi2 = res @ self.invcov @ res
            state["logp"] = -0.5 * chi2
