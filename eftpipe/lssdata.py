from __future__ import annotations
import re
import sys
import numpy as np
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Container,
    Iterable,
    Iterator,
    Pattern,
    TYPE_CHECKING,
    TypeVar,
)
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if TYPE_CHECKING:
    from numpy.typing import NDArray
from cobaya.log import HasLogger
from cobaya.log import LoggedError

_T = TypeVar("_T")


class ObsNameError(ValueError):
    pass


@dataclass(order=True, frozen=True)
class ObsName:
    symbol: str
    value: int
    latexfmt: str | None = None
    _pattern: ClassVar[Pattern] = re.compile(r"^([A-Za-z]+)(\d+)$")

    @property
    def latex(self) -> str:
        return (
            self.latexfmt.format(symbol=self.symbol, value=self.value)
            if self.latexfmt
            else f"${self.symbol}_{self.value}$"
        )

    @classmethod
    def from_str(cls, s: str) -> Self:
        match = cls._pattern.match(s)
        if not match:
            raise ObsNameError(f"{s} is not a valid ObsName")
        symbol, value = match.groups()
        return cls(symbol, int(value))

    @classmethod
    def is_valid(cls, s: str) -> bool:
        return True if cls._pattern.match(s) else False

    def clone(
        self,
        symbol: str | None = None,
        value: int | None = None,
        latexfmt: str | None = None,
    ) -> Self:
        sym = self.symbol if symbol is None else symbol
        val = self.value if value is None else value
        fmt = self.latexfmt if latexfmt is None else latexfmt
        return type(self)(sym, val, fmt)


@dataclass
class Statistics:
    x: NDArray
    y: NDArray
    yerr: NDArray | None
    name: ObsName

    def __repr__(self) -> str:
        template = "ndarray<{}, {}>([{}, ..., {}])"

        def helper(x):
            try:
                min, max = x[0], x[-1]
            except IndexError as ex:
                min, max = None, None
            return template.format(x.shape, x.dtype, min, max)

        return (
            f"{self.__class__.__name__}("
            f"x={helper(self.x)}, "
            f"y={helper(self.y)}, "
            f"yerr={None if self.yerr is None else helper(self.yerr)}, "
            f"name={self.name})"
        )


class Multipoles(Mapping, HasLogger):
    """a container for multipole-like data

    Parameters
    ----------
    x: ndarray, 1d
        k-like data in ascending order, all multipoles share the same x
    xname: str, optional
        name of x-axis, by default "k"
    **kwargs: dict[str, ndarray], 1d
        legal key names are those can be transformed into ObsName or those with suffix `error_suffix`
        legal value are 1d ndarray with the same size as x
        all names should share the same symbol
        missing yerr will be filled with None

    Attributes
    ----------
    xname: str
        name of x-axis
    yname: str
        observables' name
    x_all: ndarray, 1d
        all x data
    raw: dict[int, Statistics]
        raw data with no masking
    masked: dict[int, Statistics]
        masked data
    masks: dict[int, ndarray]
        mask for each multipole
    ls: list[int]
        masked l values
    ndata: int
        total number of masked data
    data_vector: ndarray, 1d
        flattened masked data
    data_vector_mask: ndarray, 1d
        flattened mask of masked data

    Methods
    -------
    set_mask(ls: int | Iterable[int] = (), kmin: float | Iterable[float] = (), kmax: float | Iterable[float] = ())
        mask data based on self.raw
    set_latexfmt(s: str)
        set latex format for all observables
    log_state()
        log current state
    loadtxt(path, header: Iterable[str] = None, skip: Container[str] = None) -> Self
        load multipoles from a text file

    Notes
    -----
    1. both raw and masked data are sorted in ascending order, i.e. P0, P2, P4 ...
    2. all x in `raw` and `masked` are views of the same data
    3. all y/yerr in `masked` are views of `raw`'s y/yerr
    4. data_vector and data_vector_mask owns data
    5. this class is a Mapping, one can iterate over it to access masked data
    """

    error_suffix: str = "err"

    def __init__(
        self,
        x: NDArray,
        xname: str = "k",
        logger_name="lssdata.Multipoles",
        **kwargs: NDArray,
    ) -> None:
        self.set_logger(name=logger_name)
        self.xname: str = xname

        # collect multipoles, expect same symbol and size
        if not kwargs:
            raise LoggedError(self.log, "expect at least one keyword")
        y = {}
        yerr = {}
        symbols = set()
        for k, v in kwargs.items():
            if v.size != x.size:
                raise LoggedError(self.log, "expect %s to have %d elements", k, x.size)
            adder = y
            if k.endswith(self.error_suffix):
                k = k[: -len(self.error_suffix)]
                adder = yerr
            try:
                name = ObsName.from_str(k)
            except ObsNameError as ex:
                raise LoggedError(self.log, "%s is not a valid ObsName", k) from ex
            symbols.add(name.symbol)
            adder[name.value] = v
        if len(symbols) != 1:
            raise LoggedError(
                self.log, "expect exactly one symbol, but given %s", symbols
            )
        self.yname: str = symbols.pop()
        # num of multipoles should not be less than num of errors
        ls_diff = set(yerr.keys()) - set(y.keys())
        if ls_diff:
            raise LoggedError(
                self.log, "missing %s", [f"{self.yname}{x}" for x in ls_diff]
            )

        self.raw: dict[int, Statistics] = {}
        y = {l: y[l] for l in sorted(y.keys())}  # sorted in ascending order
        x = x.copy()  # copy x only once
        self.x_all: NDArray = x
        for l, v in y.items():
            verr = yerr.get(l, None)
            verr = verr.copy() if verr is not None else None
            tmp = Statistics(x=x, y=v.copy(), yerr=verr, name=ObsName(self.yname, l))
            self.raw[l] = tmp
        self.masked: dict[int, Statistics] = {l: v for l, v in self.raw.items()}

        self.set_mask()

    def set_mask(
        self,
        ls: int | Iterable[int] = (),
        xmin: float | Iterable[float] = (),
        xmax: float | Iterable[float] = (),
        drop: bool = False,
    ) -> None:
        """apply mask filter

        ls: int | Iterable[int]
            l values to be selected, by default all
        xmin: float | Iterable[float]
            minimum x value, by default self._x_all[0] - 0.5 * (self._x_all[1] - self._x_all[0])
            if one value is given, it will be copied to match ls
        xmax: float | Iterable[float]
            maximum x value, by default self._x_all[-1] + 0.5 * (self._x_all[-1] - self._x_all[-2])
            if one value is given, it will be copied to match ls
        drop: bool, optional
            if True, masked all data, by default False
        """

        def helper(x):
            if isinstance(x, Iterable):
                out = [t for t in x]
            else:
                out = [x]
            return out

        # to list
        ls, xmin, xmax = helper(ls), helper(xmin), helper(xmax)
        self._last_set_mask = dict(
            ls=helper(ls), xmin=helper(xmin), xmax=helper(xmax), drop=drop
        )
        # default
        if not ls:
            ls = list(self.raw.keys())
        if not xmin:
            xmin = float(self.x_all[0] - 0.5 * (self.x_all[1] - self.x_all[0]))
            xmin = [xmin]
        if not xmax:
            xmax = float(self.x_all[-1] + 0.5 * (self.x_all[-1] - self.x_all[-2]))
            xmax = [xmax]
        if len(xmin) == 1:
            xmin = xmin * len(ls)
        if len(xmax) == 1:
            xmax = xmax * len(ls)
        if not (len(ls) == len(xmin) == len(xmax)):
            raise LoggedError(
                self.log,
                "expect same length, given %d ls, %d xmin, %d xmax",
                len(ls),
                len(xmin),
                len(xmax),
            )

        if not drop:
            ls_tot = [l for l in self.raw.keys()]
            ls_diff = set(ls) - set(ls_tot)
            if ls_diff:
                raise LoggedError(self.log, "unexpected ls: %s", ls_diff)
            slices = {}
            for l, xmin_, xmax_ in zip(ls, xmin, xmax):
                slices[l] = slice(
                    np.searchsorted(self.x_all, xmin_),
                    np.searchsorted(self.x_all, xmax_, side="right"),
                )
            masked = {}
            for l, v in self.raw.items():
                if l in ls:
                    sel = slices[l]
                    masked[l] = Statistics(
                        x=v.x[sel],
                        y=v.y[sel],
                        yerr=v.yerr[sel] if v.yerr is not None else None,
                        name=v.name,
                    )
            self.masked = masked
            self.ls = ls
            self.ndata = sum(v.x.size for v in self.values())
            self.data_vector = np.hstack([v.y for v in self.values()])
            xmask_bool = np.zeros(self.x_all.size, dtype=bool)
            xmasks_bool = []
            for l in ls_tot:
                sel = slices.get(l, slice(-1, 0))
                tmp = xmask_bool.copy()
                tmp[sel] = True
                xmasks_bool.append(tmp)
            self.masks = {l: mask for l, mask in zip(ls_tot, xmasks_bool)}
            self.data_vector_mask = np.hstack(xmasks_bool)
        else:
            self.masked = {}
            self.ls = []
            self.ndata = 0
            self.data_vector = np.array([])
            self.masks = {
                l: np.zeros(self.x_all.size, dtype=bool) for l in self.raw.keys()
            }
            self.data_vector_mask = np.zeros(
                self.x_all.size * len(self.ls_raw), dtype=bool
            )

    def set_latexfmt(self, s: str) -> None:
        for l, v in self.raw.items():
            name_ = v.name.clone(latexfmt=s)
            self.raw[l].name = name_
            if l in self.ls:
                self.masked[l].name = name_

    def log_state(self) -> None:
        self.mpi_info("xname=%s, yname=%s", self.xname, self.yname)
        self.mpi_info("ls=%s", self.ls)
        for l in self.ls:
            try:
                self.mpi_info(
                    f"l=%d, {self.xname}min=%.3e, {self.xname}max=%.3e, n=%d",
                    l,
                    self[l].x[0],
                    self[l].x[-1],
                    self[l].x.size,
                )
            except IndexError as ex:
                self.mpi_info("l=%d is totally masked", l)
                self.mpi_warning(
                    "Please use ls=[...] to mask specific l, "
                    "or set drop=True to mask all data",
                    l,
                )
        self.mpi_info("ndata=%d", self.ndata)

    @classmethod
    def loadtxt(
        cls,
        path,
        header: Iterable[str] | None = None,
        skip: Container[str] | None = None,
        logger_name="lssdata.Multipoles",
    ) -> Self:
        path = Path(path).resolve()
        msg = ""
        if not header:
            with path.open() as f:
                s = f.readline()
                if not s.startswith("#"):
                    msg = "no header comment, assuming 'k P0 P2 P4'"
                    header_list = ["k", "P0", "P2", "P4"]
                else:
                    header_list = s[1:].split()
        else:
            header_list = list(header)
        if not skip:
            skip = []
        data = np.loadtxt(path)
        errmsg = ""
        if len(header_list) != data.shape[-1]:
            errmsg = "header length mismatch with data"
        x = data[:, 0]
        xname = header_list.pop(0)
        dct = {name: v for name, v in zip(header_list, data.T[1:]) if name not in skip}
        dct["x"] = x
        dct["xname"] = xname
        out = cls(logger_name=logger_name, **dct)
        if errmsg:
            raise LoggedError(out.log, errmsg)
        if msg:
            out.mpi_warning(msg)
        return out

    @property
    def ls_raw(self) -> list[int]:
        return list(self.raw.keys())

    @property
    def ndata_raw(self) -> int:
        return len(self.raw) * self.x_all.size

    def __repr__(self) -> str:
        ndata = self.ndata
        ls = self.ls
        try:
            kmin = [v.x[0] for v in self.values()]
            kmax = [v.x[-1] for v in self.values()]
        except IndexError as ex:
            kmin, kmax = None, None
        return (
            f"{self.__class__.__name__}("
            f"xname={self.xname}, yname={self.yname}, "
            f"ndata={ndata}, ls={ls}, "
            f"{self.xname}min={kmin}, {self.xname}max={kmax})"
        )

    # Mapping interface
    def __getitem__(self, l: int) -> Statistics:
        return self.masked[l]

    def __iter__(self) -> Iterator[Statistics]:
        return iter(self.masked)

    def __len__(self) -> int:
        return len(self.masked)

    def keys(self):
        return self.masked.keys()

    def values(self):
        return self.masked.values()

    def items(self):
        return self.masked.items()

    def get(self, l: int, default=None) -> Statistics | None:
        return self.masked.get(l, default)


@dataclass
class BAOData:
    alperp: float
    alpara: float
    alperp_err: float | None = None
    alpara_err: float | None = None


class LSSData(HasLogger):
    """Container for Multipoles and BAOData

    Parameters
    ----------
    fullshape: Multipoles | list[Multipoles]
    cov: ndarray, 2d
        total covariance matrix, multipoles should be sorted in ascending order,
        BAO should be in this order: alperp, alpara
    bao: BAOData | list[BAOData], optional
        by default None
    cov_items: list[str], optional
        terms in covariance matrix, e.g. ['f0', 'b0', 'f1'] represents fullshape[0], bao[0], fullshape[1]
        by default ['f0', ..., 'fx']
        cov_items also determine the data_vector
    Nreal: int, optional
        number of realizations for Hartlap correction, by default 1
    rescale: float, optional
        rescale factor for covariance matrix, by default 1

    Attributes
    ----------
    fullshape: list[Multipoles]
    bao: list[BAOData]
    cov_items: list[str]
    ndata: int
        total number of masked data
    data_vector: ndarray, 1d
        flattend masked data vector
    raw_cov: ndarray, 2d
        raw unrescaled, unmasked covariance matrix
    cov: ndarray, 2d
        rescaled and masked covariance matrix
    invcov: ndarray, 2d
        rescaled, masked and hartlap-corrected inverse covariance matrix
    rescale: float
    hartlap: float | None
        hartlap correction factor

    Methods
    -------
    combine(bao_like: Iterable[T], fullshape_like: Iterable[T]) -> list[T]
        combine bao-like and fullshape-like Iterables into a list matching cov
        cached_property
    split(cov_like: Iterable[T]) -> tuple[list[T], list[T]]
        split a list which matches cov into bao-like list and fullshape-like list
        cached_property
    assign_cov(cov: ndarray)
        set yerr, alperp_err, alpara_err according to rescaled cov
    cov_mask() -> ndarray
        returns cov mask
    log_state()
        log current state
    from_dict(d: dict) -> Self
        construct from yaml/json dict
        class method
    """

    def __init__(
        self,
        fullshape: Multipoles | list[Multipoles],
        cov: NDArray,
        bao: BAOData | list[BAOData] | None = None,
        cov_items: list[str] | None = None,
        Nreal: int | None = None,
        rescale: float = 1.0,
        logger_name: str = "lssdata",
    ) -> None:
        self.set_logger(name=logger_name)

        self.fullshape = [fullshape] if isinstance(fullshape, Multipoles) else fullshape
        if bao is None:
            bao = []
        self.bao = [bao] if isinstance(bao, BAOData) else bao
        if self.bao:
            self.mpi_warning("cov assumes order in (alperp, alpara)")
        if cov_items is None:
            cov_items = [f"f{i}" for i in range(len(self.fullshape))]
        self.cov_items = cov_items
        nbao = sum(1 if x.startswith("b") else 0 for x in self.cov_items)
        nfull = sum(1 if x.startswith("f") else 0 for x in self.cov_items)
        if nbao != len(self.bao) or nfull != len(self.fullshape):
            raise LoggedError(
                self.log,
                "cov_items mismatch, "
                "please make ensure cov_items startswith 'b' or 'f'",
            )

        self.assign_cov(cov / rescale)
        self.ndata = sum(x.ndata for x in self.fullshape) + 2 * len(self.bao)
        bao_data = (np.array([x.alperp, x.alpara]) for x in self.bao)
        fullshape_data = (x.data_vector for x in self.fullshape)
        self.data_vector = np.hstack(self.combine(bao_data, fullshape_data))
        self.raw_cov = cov
        self.cov = self.raw_cov[self.cov_mask()]
        self.cov = self.cov.reshape((self.ndata, self.ndata)) / rescale
        self.rescale = rescale
        self.invcov: NDArray = np.linalg.inv(self.cov)
        self.hartlap: float | None = None
        if Nreal is not None:
            self.hartlap = (Nreal - self.ndata - 2) / (Nreal - 1)
            self.invcov *= self.hartlap

    @cached_property
    def combine(self):
        """combine bao-like and fullshape-like Iterable to a list matching cov"""
        arg = np.argsort(self.cov_items)
        unsorted_to_sorted = {j: i for i, j in enumerate(arg)}

        def out(bao_like: Iterable[_T], fullshape_like: Iterable[_T]) -> list[_T]:
            seq = list(bao_like) + list(fullshape_like)
            l = [seq[unsorted_to_sorted[i]] for i in range(len(seq))]
            return l

        return out

    @cached_property
    def split(self):
        """split a list which matches cov into bao-like list and fullshape-like list"""
        arg = np.argsort(self.cov_items)
        sorted_to_unsorted = {i: j for i, j in enumerate(arg)}
        nbao = len(self.bao)

        def out(cov_like: Iterable[_T]) -> tuple[list[_T], list[_T]]:
            seq = list(cov_like)
            l = [seq[sorted_to_unsorted[i]] for i in range(len(seq))]
            return l[:nbao], l[nbao:]

        return out

    def assign_cov(self, cov: NDArray) -> None:
        diag = np.sqrt(np.diag(cov))
        bao_ndata = [2] * len(self.bao)
        fullshape_ndata = [f.ndata_raw for f in self.fullshape]
        sections = self.combine(bao_ndata, fullshape_ndata)
        if sum(sections) != cov.shape[0]:
            raise LoggedError(self.log, "cov shape mismatch")
        errs = np.split(diag, np.cumsum(sections).tolist()[:-1])
        bao_errs, fullshape_errs = self.split(errs)
        for bao, bao_err in zip(self.bao, bao_errs):
            bao.alperp_err = bao_err[0]
            bao.alpara_err = bao_err[1]
        for fullshape, fullshape_err in zip(self.fullshape, fullshape_errs):
            for rawv, err in zip(
                fullshape.raw.values(),
                np.split(fullshape_err, len(fullshape.raw.keys())),
            ):
                rawv.yerr = err
            fullshape.set_mask(**fullshape._last_set_mask)

    def cov_mask(self) -> NDArray:
        """return mask for cov"""
        bao_masks = [np.array([True, True]) for _ in self.bao]
        fullshape_masks = [f.data_vector_mask for f in self.fullshape]
        vector_mask = np.hstack(self.combine(bao_masks, fullshape_masks))
        return np.outer(vector_mask, vector_mask)

    def log_state(self):
        self.mpi_info("total ndata: %d", self.ndata)
        self.mpi_info("fullshape items: %d", len(self.fullshape))
        if self.bao:
            self.mpi_info("BAO items: %d", len(self.bao))
        self.mpi_info("cov_items: %s", self.cov_items)
        self.mpi_info(
            "Hartlap correction: %s",
            "off" if not self.hartlap else f"{self.hartlap:.3f}",
        )
        self.mpi_info("rescale: %.3e", self.rescale)

    @classmethod
    def from_dict(cls, d: dict[str, Any], logger_name: str = "lssdata") -> Self:
        """construct from yaml/json dict"""
        d = deepcopy(d)
        bao_kwargs = d.pop("bao", [])
        bao = [BAOData(**kwargs) for kwargs in bao_kwargs]

        common = d.pop("fullshape_common", {})
        fullshape_kwargs = d.pop("fullshape", [])
        fullshape = []
        for kwargs in fullshape_kwargs:
            # copy
            kwargs_ = deepcopy(common)
            kwargs_.update(deepcopy(kwargs))
            # loadtxt
            path = kwargs_.pop("path")  # must provide
            header = kwargs_.pop("header", None)
            skip = kwargs_.pop("skip", None)
            multipole = Multipoles.loadtxt(
                path=path,
                header=header,
                skip=skip,
                logger_name=logger_name + ".Multipoles",
            )
            multipole.set_mask(**kwargs_)
            multipole.log_state()
            fullshape.append(multipole)

        cov = np.loadtxt(d.pop("cov"))
        lssdata = cls(fullshape, cov, bao, logger_name=logger_name, **d)
        lssdata.log_state()
        return lssdata
