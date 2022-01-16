# global
import numpy as np
from copy import deepcopy
from numpy import ndarray as NDArray
from typing import (
    Union,
    List,
    Dict,
    Optional,
    Any,
)
# local
from eftpipe.typing import (
    Location,
    LogFunc,
)


class PklData:
    """load power spectrum data, do k range cut

    Parameters
    ----------
    kmin: float
        min bound of kdata
    kmax: float
        max bound of kdata
    ls: int or list[int]
        ls needed, only support positive even ls
    pkl_path: str or pathlib.Path
        the path to data
    logfunc: Callable[[str], None]
        function used for logging, default print

    Notes
    -----
    kmin and kmax should be carefully chosen.
    Though this class is designed to include data within [kmin, kmax], floating number is not exact and this may not work as expected    

    Attributes
    ----------
    kdata: NDArray, float64, 1d
        range cutted ks, same for P0, P2, P4, etc...
    ls: list[int]
        sorted ls, e.g. [0, 2, 4]
    Pls: list[NDArray], float64, 1d
        list of P_l corresponding to ls
    ndata: int
        number of data points
    data_vector: NDArray, float64, 1d
        flattened Pls
    data_vector_mask: NDArray, bool, 1d
        bool mask to extract data_vector from raw flatten pkl
        used for covariance matrix
    logfunc: Callable[[str], None]
        function used for logging, default print
    _pkl: NDArray, float64, 2d
        raw data

    Raises
    ------
    TypeError:
        loaded _pkl is not a matrix
        kdata is not monotonically increasing
    ValueError:
        ls are not positive even
        _pkl doesn't contain requested ls
    """

    def __init__(
        self,
        kmin: float,
        kmax: float,
        ls: Union[int, List[int]],
        pkl_path: Location,
        logfunc: LogFunc = print,
    ) -> None:
        pkl: NDArray = np.loadtxt(pkl_path, dtype=np.float64)
        self._check_pkl(pkl)
        self._pkl = pkl

        ks = pkl[:, 0]
        kmask = slice(
            np.searchsorted(ks, kmin),
            np.searchsorted(ks, kmax, side='right')
        )
        self.kdata: NDArray = ks[kmask]

        max_nls = pkl.shape[-1] - 1
        self.ls: List[int] = self._check_ls(ls, max_nls)
        self.Pls: List[NDArray] = [pkl[:, 1 + l // 2][kmask] for l in self.ls]

        self.data_vector: NDArray = np.ravel(np.hstack(self.Pls))
        self.ndata: int = self.data_vector.size
        bool_kmask = np.zeros(ks.shape[0], dtype=bool)
        bool_kmask[kmask] = True
        all_false = np.zeros(ks.shape[0], dtype=bool)
        data_vector_mask = np.hstack(
            [(bool_kmask if (2 * i in self.ls) else all_false)
             for i in range(max_nls)]
        )
        self.data_vector_mask: NDArray = data_vector_mask

        self.logfunc: LogFunc = logfunc
        logfunc(
            "==========================>\n"
            f"loaded data from {pkl_path}\n"
            f"ls={self.ls}\n"
            f"kdata: min={self.kdata[0]:.3e}, max={self.kdata[-1]:.3e}\n"
            f"ndata={self.ndata}\n"
            "<=========================="
        )

    def _check_pkl(self, pkl: NDArray) -> None:
        if pkl.ndim != 2:
            raise TypeError(f'expect matrix, instead of ndim={pkl.ndim}')
        ks = pkl[:, 0]
        if not np.all(np.diff(ks) > 0):
            raise TypeError('expect monotonically increasing ks')

    def _check_ls(
        self,
        ls: Union[int, List[int]],
        max_nls: int
    ) -> List[int]:
        if isinstance(ls, int):
            out = [ls]
        else:
            for l in ls:
                if (l % 2 != 0) or (l < 0):
                    raise ValueError(f'invalid l={l}')
            out = sorted(ls)
        if out[-1] > (max_nls - 1) * 2:
            raise ValueError(f'pkl does not have l={out[-1]}')
        return out


class FullShapeData:
    """a container for a list of PklData and the whole covariance matrix

    Parameters
    ----------
    pkldatas: PklData or list[PklData]
        a list of PklData objects, or just a single PklData
    cov_path: str or pathlib.Path
        path to covariance matrix
        it should fully cover the list of PklData._pkl
    Nreal: int or None
        number of realizations, used to apply Hartlap factor to the inverse of covariance matrix
        default None and no correction is applied
    rescale: float
        rescale factor, which will be mutipied to the inverse of covariance matrix
        default 1.0
    logfunc: Callable[[str], None]
        function used for logging, default print

    Attributes
    ----------
    pkldatas: list[PklData]
        a list of PklData objects
    ndata: int
        number of data points
    data_vector: NDArray, float64, 1d
        the whole data vector
    cov: NDArray, float64, 2d
        the covariance matrix of requested data only
    invcov: NDArray, float64, 2d
        the inverse of covariance matrix, rescaled and corrected by Hartlap factor
    logfunc: Callable[[str], None]
        function used for logging, default print
    _cov: NDArray, float64, 2d
        the whole covariance matrix

    Raises
    ------
    TypeError:
        loaded _cov is not a square matrix
    ValueError:
        _cov and pkldatas not match
    """

    def __init__(
        self,
        pkldatas: Union[PklData, List[PklData]],
        cov_path: Location,
        Nreal: Optional[int] = None,
        rescale: float = 1.0,
        logfunc: LogFunc = print,
    ) -> None:
        self.pkldatas: List[PklData] = (
            pkldatas if isinstance(pkldatas, list) else [pkldatas]
        )
        self.ndata: int = sum(pkldata.ndata for pkldata in self.pkldatas)
        self.data_vector: NDArray = np.ravel(
            np.hstack([pkldata.data_vector for pkldata in self.pkldatas])
        )
        assert self.data_vector.shape[0] == self.ndata

        _cov = np.loadtxt(cov_path, dtype=np.float64)
        self._check_cov(_cov)
        self._cov: NDArray = _cov
        data_vector_mask = np.ravel(
            np.hstack([_.data_vector_mask for _ in self.pkldatas]))
        cov = _cov[np.outer(data_vector_mask, data_vector_mask)].reshape(
            (self.ndata, -1))
        self.cov: NDArray = cov
        self.invcov: NDArray = np.linalg.inv(cov) * rescale
        if Nreal is not None:
            self.invcov *= (Nreal - self.ndata - 2) / (Nreal - 1)
        assert self.invcov.shape[0] == self.ndata

        self.logfunc: LogFunc = logfunc
        logfunc(
            "==========================>\n"
            f"total ndata={self.ndata}\n"
            f"Hartlap correction: {'on' if (Nreal is not None) else 'off'}\n"
            f"rescale factor: {rescale:.3e}\n"
            "<=========================="
        )

    def _check_cov(self, cov: NDArray) -> None:
        ndim = cov.ndim
        if ndim != 2:
            raise TypeError(f'expect matrix, instead of ndim={ndim}')
        shape = cov.shape
        if shape[0] != shape[-1]:
            raise TypeError(f'expect square matrix, instead of shape={shape}')
        nall = sum([_.data_vector_mask.shape[0] for _ in self.pkldatas])
        if shape[0] != nall:
            raise ValueError('pkl and cov not match')


class FullShapeDataParser:
    """a factory to create FullShapeData object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create FullShapeData
    logfunc: Callable[[str], None]
        function used for logging, default print

    Methods
    -------
    helper_dict(cls):
        a template to create dict
    create_gaussian_data(self):
        create a FullShapeData object
    """

    def __init__(
        self,
        dct: Dict[str, Any],
        logfunc: LogFunc = print,
    ) -> None:
        dct = deepcopy(dct)
        pklinfo = dct.pop('pklinfo')
        if not isinstance(pklinfo, list):
            pklinfo = [pklinfo]

        common = dct.pop('common', None)
        if common is not None:
            new_pklinfo = [deepcopy(common) for _ in pklinfo]
            for raw, new in zip(new_pklinfo, pklinfo):
                raw.update(new)
            pklinfo = new_pklinfo

        dct['pkldatas'] = [PklData(**x, logfunc=logfunc) for x in pklinfo]
        self.dct = dct
        self.logfunc = logfunc

    @classmethod
    def helper_dict(cls):
        return {
            "cov_path": None,
            "Nreal": None,
            "rescale": None,
            "common": {"kmin": None, "kmax": None, "ls": None},
            "pklinfo": [
                {"pkl_path": None},
                {"pkl_path": None},
            ]
        }

    def create_gaussian_data(self) -> FullShapeData:
        return FullShapeData(logfunc=self.logfunc, **self.dct)
