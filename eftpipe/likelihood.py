from __future__ import annotations
import importlib
import logging
import re
import numpy as np
from typing import Any, Iterable, List, Union
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood
from .marginal import Marginalizable
from .reader import read_pkl
from .tools import int_or_list


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


FloatBound_T = Union[float, List[float], None]


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

    def regularize(x: FloatBound_T, default: float) -> list[float]:
        if x is None:
            return [default] * nl
        elif isinstance(x, float):
            return [x] * nl
        else:
            if len(x) != nl:
                raise ValueError(f"length of kmin/kmax mask not matching ells")
            return list(x)

    ret: dict[int, slice] = {}
    kminlst = regularize(kmin, -1)
    kmaxlst = regularize(kmax, 1e10)
    for ell, min_kmask, max_kmask in zip(sorted(ells), kminlst, kmaxlst):
        ileft = np.searchsorted(kall, min_kmask)
        iright = np.searchsorted(kall, max_kmask, side="right")
        ret[ell] = slice(ileft, iright)
    return ret


def slice_union(slices: Iterable[slice]) -> slice:
    start = min(s.start for s in slices)
    stop = max(s.stop for s in slices)
    return slice(start, stop)


def mask_covariance(
    cov, ls: list[int], ls_tot: list[int], kall, kmin: FloatBound_T, kmax: FloatBound_T
):
    """generate masked covariance matrix

    Parameters
    ----------
    cov : ndarray, 2d
        covariance matrix
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
    kmask = parse_kmask(kall, ls, kmin, kmax)
    nl, nk = len(ls_tot), len(kall)
    if cov.shape[0] != nl * nk:
        raise ValueError(
            f"covariance matrix's shape {cov.shape} does not match input data, "
            f"expected {(nl * nk, nl * nk)}"
        )
    mask = np.zeros((nl, nk), dtype=bool)
    for i, ell in enumerate(ls_tot):
        mask[i, kmask.get(ell, [])] = True
    return cov[np.outer(mask1d := mask.flatten(), mask1d)].reshape(mask1d.sum(), -1)


def hartlap(Nreal: int, ndata: int) -> float:
    return (Nreal - ndata - 2) / (Nreal - 1)


def flatten(ls: list[int], array, mask: dict[int, slice] | None = None):
    """

    Parameters
    ----------
    ls : list[int]
        multipole order, should be even
    array : ndarray, 2d
    mask : dict[int, slice] | None
    """
    idx = [ell // 2 for ell in ls]
    if not mask:
        return array[idx, :].flatten()
    return np.hstack([array[i, mask[ell]] for i, ell in zip(idx, ls)])


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

        self.df = df = find_data_reader(
            self.data.get("reader", "auto"),
            self.data.get("reader_kwargs", {}),
            self.log,
        )(self.data["path"])
        symbol, ls_tot = extract_multipole_info(df.columns)
        self.ls_tot = ls_tot
        self.ls = ls = int_or_list(self.data["ls"])
        if not_exsited_ls := set(ls).difference(ls_tot):
            raise ValueError(f"ls {not_exsited_ls} not found in data")
        kmask = parse_kmask(df.index, ls, self.data.get("kmin"), self.data.get("kmax"))
        self.data_vector = np.hstack(
            [df[symbol + str(ell)].to_numpy()[kmask[ell]] for ell in ls]
        )
        self.ndata = self.data_vector.size
        # kout is the k-grid which the theory code will compute on
        self.kout = kout = df.index.to_numpy()[slice_union(kmask.values())]
        # redefine kmask
        self.kmask = kmask = parse_kmask(
            kout, ls, self.data.get("kmin"), self.data.get("kmax")
        )
        self.binning = self.binning or {}
        self.binning = {"kout": kout, **self.binning}

        self.set_invcov()
        self.log_data_loading()

    def set_invcov(self) -> None:
        if not isinstance(self.cov, dict):
            self.cov = {"path": self.cov}
        cov = find_covariance_reader(
            self.cov.get("reader", "auto"), self.cov.get("reader_kwargs", {})
        )(self.cov["path"])
        cov = mask_covariance(
            cov,
            self.ls,
            self.ls_tot,
            self.df.index,
            self.data.get("kmin"),
            self.data.get("kmax"),
        )
        cov /= self.cov.get("rescale", 1)
        self.invcov = np.linalg.inv(cov)
        self.hartlap: float | None = None
        if (Nreal := self.cov.get("Nreal")) is not None:
            self.hartlap = hartlap(Nreal, self.ndata)
            self.invcov *= self.hartlap

    def log_data_loading(self) -> None:
        self.mpi_info("data ells=%s", self.ls)
        for ell in self.ls:
            kmasked = self.kout[self.kmask[ell]]
            self.mpi_info(
                f"ell=%d, kmin=%.3e, kmax=%.3e, ndata=%d",
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

    def get_requirements(self):
        reqs = {}
        if self.with_binning:
            reqs["nonlinear_Plk_grid"] = {
                self.tracer: {
                    "ls": self.ls,
                    "chained": self.chained,
                    "binned": self.with_binning,
                    "binning": self.binning,
                }
            }
        else:
            reqs["nonlinear_Plk_interpolator"] = {
                self.tracer: {
                    "ls": self.ls,
                    "chained": self.chained,
                }
            }
        if self.marg:
            reqs["nonlinear_Plk_gaussian_grid"] = {
                self.tracer: {
                    "ls": self.ls,
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
        ret = []
        for bG in self.valid_prior.keys():
            plk = table[bG]
            if not self.with_binning:
                interpfn = interp1d(kgrid, kgrid * plk, kind="cubic", axis=-1)
                fn = lambda k: interpfn(k) / k
                plk = fn(self.kout)
            ret.append(flatten(self.ls, plk, self.kmask))
        return np.vstack(ret)

    # override
    def PNG(self):
        if self.with_binning:
            _, _, plk = self.provider.get_nonlinear_Plk_grid(
                self.tracer, chained=self.chained, binned=self.with_binning
            )
            ret = flatten(self.ls, plk, self.kmask)
        else:
            fn = self.provider.get_nonlinear_Plk_interpolator(
                self.tracer, chained=self.chained
            )
            ret = flatten(self.ls, fn(self.ls, self.kout), self.kmask)
        return ret

    def calculate(self, state, want_derived=True, **params_values_dict):
        if self.marg:
            state["logp"] = self.marginalized_logp(self.data_vector, self.invcov)
        else:
            res = self.data_vector - self.PNG()
            chi2 = res @ self.invcov @ res
            state["logp"] = -0.5 * chi2

    def bG_bestfit(self) -> dict[str, float]:
        """helper method to extract bestfit bG parameters"""
        PNG = self.PNG()
        PG = self.PG()
        F1i = self.calc_F1i(PG, PNG, self.invcov, self.data_vector)
        F2ij = self.calc_F2ij(PG, self.invcov)
        ret = np.linalg.inv(F2ij) @ F1i
        return {bG: val for bG, val in zip(self.valid_prior.keys(), ret)}
