from __future__ import annotations
import importlib
import itertools
import logging
import re
import numpy as np
import pandas as pd
import scipy
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, List, TYPE_CHECKING, Union
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood
from .marginal import Marginalizable
from .marginal import valid_prior_config
from .reader import read_pkl
from .tools import int_or_list
from .tools import str_or_list
from .tools import pairwise

if TYPE_CHECKING:
    from .parambasis import EFTBasis

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
    elif isinstance(x, float) or isinstance(x, int):
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
    *args : ``ls``, ``ls_tot``, ``kall``, ``kmin``, ``kmax`` in order
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


def regularize_prior(prior: dict[str, dict[str, Any]]):
    # support additional prefix form, e.g.
    # ELG_NGC_b1:
    #   loc: 0
    #   scale: 1
    # LRG_NGC:
    #   b1:
    #     loc: 0
    #     scale: 1
    #   b2:
    #     loc: 0
    #     scale: 1
    # LRG_SGC:
    #   b1:
    #   ...
    processed: dict[str, dict[str, Any]] = {}
    for p, config in prior.items():
        if valid_prior_config(config):
            processed[p] = config
            continue
        if isinstance(config, dict):
            config = deepcopy(config)
            for param, subconfig in config.items():
                processed[f"{p}{param}"] = subconfig
        else:
            raise ValueError(f"invalid prior config: {config}")
    return processed


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
        symbol, ls_tot = extract_multipole_info(df.columns.to_list())
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


class EFTLike(Likelihood, Marginalizable):
    """EFT likelihood for arbitrary number of tracers and cross-correlations."""

    file_base_name = "eftlike"
    # fmt: off
    likelihood_prefix: str
    marg_param_prefix: str
    tracers: list[str]  # also support str
    data: dict[str, dict[str, Any]]  # also support dict[str, Any] | list[dict[str, Any]]
    cov: dict[str, Any]  # also support str for path
    chained: dict[str, bool]  # also support bool | list[bool]
    with_binning: dict[str, bool]  # also support bool | list[bool]
    binning: dict[str, dict[str, Any]]  # also support dict[str, Any] | list[dict[str, Any]]
    marg: dict[str, dict[str, Any]]
    # fmt: on

    def initialize(self) -> None:
        super().initialize()
        if self.likelihood_prefix is None:
            self.likelihood_prefix = self.get_name() + "_"
        self.regularize_attributes()
        self.minfodict = {
            t: MultipoleInfo.load(**self.data[t], logger=self.log) for t in self.tracers
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
        self.tracers = str_or_list(self.tracers)
        ntracer = len(self.tracers)

        if ntracer == 1:

            def to_nested_dict(x: Any) -> Any:
                if isinstance(x, dict) and [*x.keys()][0] != self.tracers[0]:
                    return {self.tracers[0]: x}
                return x

            self.data = to_nested_dict(self.data)
            self.binning = to_nested_dict(self.binning)

        def or_list_to_dict(x: Any) -> Any:
            if not isinstance(x, (dict, list)):
                x = [x] * ntracer
            if isinstance(x, list):
                return {t: x[i] for i, t in enumerate(self.tracers)}
            return x

        self.data: dict[str, dict[str, Any]] = or_list_to_dict(self.data)
        self.chained: dict[str, bool] = or_list_to_dict(self.chained)
        self.with_binning: dict[str, bool] = or_list_to_dict(self.with_binning)
        self.binning = self.binning or {t: {} for t in self.tracers}
        self.binning: dict[str, dict[str, Any]] = or_list_to_dict(self.binning)

    def set_invcov(self) -> None:
        if not isinstance(self.cov, dict):
            self.cov = {"path": self.cov}
        reader = find_covariance_reader(
            self.cov.get("reader", "auto"), self.cov.get("reader_kwargs", {})
        )
        if isinstance(path := self.cov["path"], list):
            cov = scipy.linalg.block_diag(*[reader(p) for p in path])
        else:
            cov = reader(path)
        args: Any = ()
        for minfo in self.minfodict.values():
            args += (
                minfo.ls,
                minfo.ls_tot,
                minfo.df.index,
                minfo.kmin,
                minfo.kmax,
            )
        cov /= self.cov.get("rescale", 1)
        self.hartlap: float | None = None
        if (Nreal := self.cov.get("Nreal")) is not None:
            self.hartlap = hartlap(Nreal, self.ndata)
            cov /= self.hartlap
        self.full_covmat = cov
        cov = mask_covariance(cov, *args)
        self.invcov = np.linalg.inv(cov)

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

    def get_requirements(self):
        reqs = {
            "nonlinear_Plk_grid": {},
            "nonlinear_Plk_interpolator": {},
            "nonlinear_Plk_gaussian_grid": {},
        }
        for t, minfo, chained, with_binning, binning in zip(
            self.tracers,
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
            # and it's not possible to find out which tracer is marginalized given ``self.marg``
            if self.marg:
                reqs["nonlinear_Plk_gaussian_grid"][t] = {
                    "ls": minfo.ls,
                    "chained": chained,
                    "binned": with_binning,
                    **({"binning": binning} if with_binning else {}),
                }
                reqs.setdefault("eft_params_values_dict", {})
                reqs["eft_params_values_dict"][t] = None
        reqs = {k: v for k, v in reqs.items() if v}
        return reqs

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)
        self.eft_bases: list[EFTBasis] = [
            self.provider.model.theory["eftpipe.eftlss." + t].basis
            for t in self.tracers
        ]
        # setup prior after prefix is set
        if self.marg:
            self.setup_prior(self.marg)
            self.report_marginalized()
        self.set_cache()

    def set_cache(self) -> None:
        self._PNG_cache = np.zeros(self.data_vector.size)
        sizelist: list[int] = [
            minfo.data_vector.size for minfo in self.minfodict.values()
        ]
        self._istart_iend_cache: list[tuple[int, int]] = list(
            pairwise(itertools.accumulate(sizelist, initial=0))
        )
        if self.marg:
            bG_group: list[list[str]] = [list() for _ in self.tracers]
            to_assign = self.valid_prior.keys()
            for bGlist, basis in zip(bG_group, self.eft_bases):
                allowed = basis.gaussian_params()
                # use for-loop to preserve the order
                for name in to_assign:
                    if name in allowed:
                        bGlist.append(name)
            self._bG_group_cache = bG_group
            counter = 0
            bGidx: dict[str, int] = {}
            for name in itertools.chain(*bG_group):
                if name not in bGidx:
                    bGidx[name] = counter
                    counter += 1
            self._bGidx_cache = bGidx
            self._PG_cache = np.zeros((len(bGidx), self.data_vector.size))

    # impl
    def marginalizable_params(self) -> list[str]:
        params = []
        for basis in self.eft_bases:
            params += basis.gaussian_params()
        return list(dict.fromkeys(params))

    # impl
    def PG(self):
        bG_idx = self._bGidx_cache
        for tracer, minfo, chained, with_binning, bGlist, (istart, iend) in zip(
            self.tracers,
            self.minfodict.values(),
            self.chained.values(),
            self.with_binning.values(),
            self._bG_group_cache,
            self._istart_iend_cache,
        ):
            # TODO: ells trim
            _, kgrid, table = self.provider.get_nonlinear_Plk_gaussian_grid(
                tracer,
                chained=chained,
                binned=with_binning,
            )
            for bG in bGlist:
                plk = table[bG]
                if not with_binning:
                    interpfn = interp1d(kgrid, kgrid * plk, kind="cubic", axis=-1)
                    fn = lambda k: interpfn(k) / k
                    plk = fn(minfo.kout)
                flatten(
                    minfo.ls,
                    plk,
                    minfo.kout_mask,
                    out=self._PG_cache[bG_idx[bG], istart:iend],
                )
        # be careful, return a reference
        return self._PG_cache

    # impl
    def PNG(self) -> Any:
        for t, minfo, with_binning, chained, (istart, iend) in zip(
            self.tracers,
            self.minfodict.values(),
            self.with_binning.values(),
            self.chained.values(),
            self._istart_iend_cache,
        ):
            # TODO: ells trim
            if with_binning:
                _, _, plk = self.provider.get_nonlinear_Plk_grid(
                    t, chained=chained, binned=True
                )
            else:
                fn = self.provider.get_nonlinear_Plk_interpolator(t, chained=chained)
                plk = fn(minfo.ls, minfo.kout)
            flatten(minfo.ls, plk, minfo.kout_mask, out=self._PNG_cache[istart:iend])
        # be careful, return a reference
        return self._PNG_cache

    # impl
    def get_data_vector(self):
        return self.data_vector

    # impl
    def get_invcov(self):
        return self.invcov

    # override
    def env(self):
        retval = {"np": np}
        for t in self.tracers:
            retval.update(self.provider.get_eft_params_values_dict(t))
        return retval

    # override
    def update_prior(
        self, prior: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float | str]]:
        return super().update_prior(regularize_prior(prior))

    def calculate(self, state, want_derived=True, **params_values_dict):
        return_bGbest = True if self.required_bGbest_related_derived_params() else False
        fullchi2, bGbest = -1, defaultdict(lambda: 0.0)
        if self.marg:
            tmp = self.marginalized_logp(return_bGbest=return_bGbest)
            if isinstance(tmp, tuple):
                logp, fullchi2, bGbest = tmp
            else:
                logp = tmp
            state["logp"] = logp
            # in this case, the chi2 is meaningless...
            chi2 = -2 * logp
        else:
            res = self.data_vector - self.PNG()
            chi2 = res @ self.invcov @ res
            fullchi2 = chi2
            state["logp"] = -0.5 * chi2
        if want_derived:
            state["derived"][self.likelihood_prefix + "chi2"] = chi2
            state["derived"][self.likelihood_prefix + "fullchi2"] = fullchi2
            if return_bGbest:
                for p in self.output_params:
                    if p.startswith(self.marg_param_prefix):
                        nlen = len(self.marg_param_prefix)
                        state["derived"][p] = bGbest[p[nlen:]]

    def get_can_provide_params(self) -> list[str]:
        return [
            self.likelihood_prefix + p for p in ("chi2", "fullchi2")
        ] + self.estimated_can_provide_params()

    def estimated_can_provide_params(self) -> list[str]:
        retval = []
        if marg := self.marg:
            for p, config in marg.items():
                if valid_prior_config(config):
                    retval.append(self.marg_param_prefix + p)
                if isinstance(config, dict):
                    for param, subconfig in config.items():
                        retval.append(self.marg_param_prefix + p + param)
                else:
                    raise ValueError(f"invalid prior config: {config}")
        return retval

    def required_bGbest_related_derived_params(self) -> bool:
        return any(p.startswith(self.marg_param_prefix) for p in self.output_params)
