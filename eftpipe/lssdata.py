import re
import sys
import numpy as np
from copy import deepcopy
from functools import total_ordering
from pathlib import Path
from typing import (
    Any,
    cast,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from numpy import ndarray as NDArray
from cobaya.log import HasLogger

_T = TypeVar("_T")


class ObsNameError(ValueError):
    pass


@total_ordering
class ObsName:
    _pattern = re.compile(r"^([A-Za-z]+)(\d+)$")

    def __init__(
        self, symbol: str, value: int, latexfmt: Optional[str] = None
    ) -> None:
        self.symbol = symbol
        self.value = value
        self._latex = latexfmt

    @property
    def latex(self) -> str:
        return (
            self._latex.format(symbol=self.symbol, value=self.value)
            if self._latex
            else f"{self.symbol}_{self.value}"
        )

    @classmethod
    def from_str(cls, s: str) -> "ObsName":
        match = cls._pattern.match(s)
        if match:
            symbol, value = match.groups()
        else:
            raise ObsNameError(f"{s} is invalid ObsName")
        return cls(symbol, int(value))

    def clone(
        self,
        symbol: Optional[str] = None,
        value: Optional[int] = None,
        latexfmt: Optional[str] = None
    ) -> "ObsName":
        sym = self.symbol if symbol is None else symbol
        val = self.value if value is None else value
        return self.__class__(sym, val, latexfmt)

    @classmethod
    def validq(cls, s: str) -> bool:
        return True if cls._pattern.match(s) else False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"symbol={self.symbol}, "
            f"value={self.value}"
            f")"
        )

    def __str__(self) -> str:
        return self.symbol + str(self.value)

    def __hash__(self) -> int:
        return hash((self.symbol, self.value, self._latex))

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def __eq__(self, other) -> bool:
        return str(self) == str(other)


class DistinctError(ValueError):
    pass


class Same:
    def __init__(self, obj: object, key: Callable) -> None:
        self.obj = obj
        self.base = key(obj)
        self.key = key

    def validate(self, obj: _T) -> _T:
        try:
            v = self.key(obj)
            if self.base == v:
                return obj
        except Exception as e:
            raise DistinctError(f"cannot apply {self.key!r} to {obj!r}") from e
        else:
            raise DistinctError(f"distinct result from {self.obj!r}")


class MaskTuple(NamedTuple):
    ls: List[int]
    kmin: float
    kmax: float


class RawPklData:
    def __init__(
        self,
        kdata: NDArray,
        data: Dict[ObsName, NDArray],
        err: Dict[ObsName, Optional[NDArray]]
    ) -> None:
        self.kdata = kdata
        self.data = data
        self.err = err
        self.ndata = sum(x.size for x in self.data.values())
        self.ls = [k.value for k in self.data.keys()]

    def set_latexfmt(self, s: str) -> None:
        self.data = {k.clone(latexfmt=s): v for k, v in self.data.items()}
        self.err = {k.clone(latexfmt=s): v for k, v in self.err.items()}


class PklData(HasLogger):
    """a container for pkl data and err

    Parameters
    ----------
    kdata: ndarray, 1d
    **kwargs: dict[str, ndarray]
        legal key names are those can be transformed into ObsName or those with suffix `error_suffix`,
        legal value are 1d ndarray

    Attributes
    ----------
    symbol: str
        symbol name of data
    kdata: ndarray, 1d
    raw: RawPklData
        container for raw data
    data_vector: ndarray, 1d
        masked flattened data
    data_vector_mask: ndarray, 1d, bool
        data mask
    ndata: int
        total number of masked data
    ls: list[int]
        contained multipoles, sorted
    mask: MaskTuple
        last mask settings

    Methods
    -------
    set_mask(ls: list[int] | int = None, kmin: float = None, kmax: float = None)
        mask data based on `self.raw`
    set_latexfmt(s: str)
        set latex format, eg. "{symbol}_{value}"
    log_state()
        log current state
    loadtxt(path, header: Iterable[str] = None, skip: Container[str] = None)
        classmethod, create PklData from txt
        If txt doesn't have header, assuming "kdata P0 P2 P4"
        'k' is an alias of 'kdata'

    Notes
    -----
    both data and err are sorted in ascending order, i.e. P0, P2, P4 ...
    """
    error_suffix: str = 'err'

    def __init__(self, kdata: NDArray, **kwargs: NDArray) -> None:
        """init

        Raises
        ------
        ValueError
            no kwargs provided
        ObsNameError
            illegal key name
        DistinctError
            data.shape != kdata.shape
        """
        self.symbol: str = ""
        self.kdata = kdata
        if not kwargs:
            raise ValueError('expect at least one keyword')
        self._set_data_and_err(**kwargs)
        self.raw = RawPklData(self.kdata, self.data, self.err)
        self.data_vector = np.hstack([v for v in self.data.values()])
        self.data_vector_mask = np.ones(self.data_vector.size, dtype=bool)
        self.ndata = self.data_vector.size
        self.ls = [v.value for v in self.data.keys()]
        self.mask = MaskTuple(
            ls=self.ls,
            kmin=self.kdata.min() - 0.1,
            kmax=self.kdata.max() + 0.1
        )
        self.set_logger(name="lssdata.PklData")

    def _set_data_and_err(self, **kwargs: NDArray) -> None:
        data: Dict[ObsName, NDArray] = {}
        err: Dict[ObsName, Optional[NDArray]] = {}
        sameshape = Same(self.kdata, len)
        for obsname, value in kwargs.items():
            todata = True
            if obsname.endswith(self.error_suffix):
                obsname = obsname[:-len(self.error_suffix)]
                todata = False
            obsname = ObsName.from_str(obsname)
            if not self.symbol:
                self.symbol = obsname.symbol
            else:
                if obsname.symbol != self.symbol:
                    raise ObsNameError(f"unexpected obsname {obsname}")
            if todata:
                data[obsname] = sameshape.validate(value)
            else:
                err[obsname] = sameshape.validate(value)
        data = dict(sorted(data.items()))
        # obsname in err should appear in data
        for obsname in err.keys():
            if obsname not in data.keys():
                raise ObsNameError(f"unexpected obsname{self.error_suffix}")
        # default None if not set
        for obsname, value in data.items():
            res = err.get(obsname, None)
            if res is None:
                err[obsname] = None
        err = dict(sorted(err.items()))
        self.data, self.err = data, err

    def set_mask(
        self,
        ls: Optional[Union[int, List[int]]] = None,
        kmin: Optional[float] = None,
        kmax: Optional[float] = None,
    ) -> None:
        if ls is None:
            ls = self.raw.ls
        elif isinstance(ls, int):
            ls = [ls]
        kmin = self.raw.kdata.min() - 1.0 if kmin is None else kmin
        kmax = self.raw.kdata.max() + 1.0 if kmax is None else kmax
        kmin, kmax = cast(float, kmin), cast(float, kmax)
        kmask = slice(
            np.searchsorted(self.raw.kdata, kmin),
            np.searchsorted(self.raw.kdata, kmax, side='right')
        )
        self.mask = MaskTuple(ls=ls, kmin=kmin, kmax=kmax)
        newdata: Dict[ObsName, NDArray] = {}
        newerr: Dict[ObsName, Optional[NDArray]] = {}
        for name, arr in self.raw.data.items():
            if name.value in ls:
                newdata[name] = arr[kmask]
        for name, arr in self.raw.err.items():
            if name.value in ls:
                newerr[name] = arr[kmask] if arr is not None else None
        self.kdata = self.raw.kdata[kmask]
        self.data = newdata
        self.err = newerr
        self.ls = [k.value for k in newdata.keys()]
        self.ndata = sum(v.size for v in newdata.values())
        self.data_vector = np.hstack([v for v in newdata.values()])
        kmask_bool = np.zeros(self.raw.kdata.size, dtype=bool)
        kmask_bool[kmask] = True
        allfalse = np.zeros(self.raw.kdata.size, dtype=bool)
        self.data_vector_mask = np.hstack(
            [(kmask_bool if l in self.ls else allfalse) for l in self.raw.ls]
        )

    def set_latexfmt(self, s: str) -> None:
        self.data = {k.clone(latexfmt=s): v for k, v in self.data.items()}
        self.err = {k.clone(latexfmt=s): v for k, v in self.err.items()}
        self.raw.set_latexfmt(s)

    def log_state(self) -> None:
        self.mpi_info("symbol=%s", self.symbol)
        self.mpi_info("ls=%s", self.ls)
        self.mpi_info(
            "kdata: min=%.3e, max=%.3e", self.kdata[0], self.kdata[-1])
        self.mpi_info("ndata=%d", self.ndata)

    @classmethod
    def loadtxt(
        cls,
        path,
        header: Optional[Iterable[str]] = None,
        skip: Optional[Container[str]] = None,
        log: bool = True,
    ) -> "PklData":
        path = Path(path).resolve()
        msg = ""
        if not header:
            with path.open() as f:
                s = f.readline()
                if not s.startswith('#'):
                    msg = "no header comment, assuming 'kdata P0 P2 P4'"
                    header_list = ['kdata', 'P0', 'P2', 'P4']
                else:
                    header_list = s.split()[1:]
        else:
            header_list = list(header)
        # alias
        if 'k' in header_list:
            i = header_list.index('k')
            header_list[i] = 'kdata'
        if not skip:
            skip = []
        data = np.loadtxt(path)
        dct = {
            name: v for name, v in zip(header_list, data.T)
            if name not in skip
        }
        out = cls(**dct)
        if msg and log:
            out.mpi_warning(msg)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"symbol={self.symbol}, "
            f"ls={self.ls}, "
            f"kmin={self.kdata[0]:.4f}, "
            f"kmax={self.kdata[-1]:.4f}, "
            f"ndata={self.ndata}"
            f")"
        )

    def __getattr__(self, name: str) -> Optional[NDArray]:
        try:
            if name.endswith(self.error_suffix):
                key = ObsName.from_str(name[:-len(self.error_suffix)])
                return self.err[key]
            else:
                key = ObsName.from_str(name)
                return self.data[key]
        except (ObsNameError, KeyError) as ex:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            ) from ex


class PklDataDict(TypedDict, total=False):
    pkl_path: str
    ls: List[int]
    kmin: float
    kmax: float
    header: List[str]
    skip: List[str]


class FullShapeDataDict(TypedDict, total=False):
    """dictionary for FullShapeData construction"""
    pklinfo: List[PklDataDict]
    common: PklDataDict
    cov_path: str
    Nreal: int
    rescale: float


class FullShapeData(Sequence[PklData], HasLogger):
    """Sequence Container for PklData and covariance matrix

    Parameters
    ----------
    pkldatas: PklData | Iterable[Pkldata]
    cov: ndarray, 2d
        covariance matrix for pkldatas, multipoles should be sorted in ascending order
    Nreal: int
        realizations, for Hartlap correction
    rescale: float
        rescale the covariance matrix, can be understand as rescale x survey volume

    Attributes
    ----------
    pkldatas: list[PklData]
    ndata: int
    data_vector: ndarray, 1d
        flattend masked data vector
    raw_cov: ndarray, 2d
        raw unrescaled, unmasked covariance matrix
    cov: ndarray, 2d
        rescaled and masked covariance matrix
    invcov: ndarray, 2d
        rescaled, masked and hartlap corrected covariance matrix inverse
    rescale: float
    hartlap: float | None
        hartlap factor

    Methods
    -------
    log_state()
        log current state
    from_dict(dct: FullShapeDataDict)
        classmethod, construct FullShapeData from dictionary
    """

    def __init__(
        self,
        pkldatas: Union[PklData, Iterable[PklData]],
        cov: NDArray,
        Nreal: Optional[int] = None,
        rescale: float = 1.0,
    ) -> None:
        self.pkldatas = (
            [pkldatas] if isinstance(pkldatas, PklData) else list(pkldatas)
        )
        self.assign_cov(self.pkldatas, cov / rescale)
        self.ndata: int = sum(x.ndata for x in self)
        self.data_vector = np.hstack([x.data_vector for x in self])
        self.raw_cov = cov
        data_vector_mask = np.hstack(
            [x.data_vector_mask for x in self])
        self.cov = cov[np.outer(data_vector_mask, data_vector_mask)]
        self.cov: NDArray = self.cov.reshape(
            (self.ndata, self.ndata)) / rescale
        self.rescale = rescale
        self.invcov: NDArray = np.linalg.inv(self.cov)
        self.hartlap: Optional[float] = None
        if Nreal is not None:
            self.hartlap = (Nreal - self.ndata - 2) / (Nreal - 1)
            self.invcov *= self.hartlap
        self.set_logger(name="lssdata.FullShapeData")

    def assign_cov(self, pkldatas: List[PklData], cov: NDArray) -> None:
        diag = np.sqrt(np.diag(cov))
        start = 0
        for pkldata in pkldatas:
            nraw = pkldata.raw.ndata
            nl = len(pkldata.raw.ls)
            newerrs = diag[start:start + nraw].reshape((nl, -1))
            for (obsname, _), err in zip(pkldata.raw.err.items(), newerrs):
                pkldata.raw.err[obsname] = err
            mask = pkldata.mask
            pkldata.set_mask(ls=mask.ls, kmin=mask.kmin, kmax=mask.kmax)
            start += nraw

    def log_state(self) -> None:
        self.mpi_info("total ndata=%d", self.ndata)
        self.mpi_info("PklData items: %d", self.__len__())
        self.mpi_info(
            "Hartlap correction: %s",
            'off' if not self.hartlap else f"{self.hartlap:.3f}"
        )
        self.mpi_info('rescale factor: %.3e', self.rescale)

    @classmethod
    def from_dict(
        cls, dct: FullShapeDataDict, log: bool = True
    ) -> "FullShapeData":
        cov: NDArray = np.loadtxt(dct['cov_path'])  # type: ignore
        Nreal = dct.get("Nreal")
        rescale = dct.get("rescale", 1.0)
        common = dct.get("common", {})
        pklinfo = dct['pklinfo']  # type: ignore
        if not isinstance(pklinfo, list):
            pklinfo = [pklinfo]
        pklinfo = cast(List[PklDataDict], pklinfo)
        pkldatas: List[PklData] = []
        for info in pklinfo:
            default = cast(Dict[str, Any], deepcopy(common))
            for k, v in info.items():
                default[k] = v
            item = PklData.loadtxt(
                default['pkl_path'],
                header=default.get('header'), skip=default.get('skip'), log=log,
            )
            item.set_mask(
                ls=default.get('ls'),
                kmin=default.get('kmin'),
                kmax=default.get('kmax')
            )
            if log:
                item.log_state()
            pkldatas.append(item)
        out = cls(pkldatas, cov, Nreal, rescale)
        if log:
            out.log_state()
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}["
            + ", ".join([x.__repr__() for x in self]) +
            f"]"
        )

    def __getitem__(self, i) -> PklData:
        return self.pkldatas[i]

    def __len__(self) -> int:
        return len(self.pkldatas)
