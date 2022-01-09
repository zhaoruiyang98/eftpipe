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
from eftpipe.typing import Location


class CroppedPklData:
    kdata: NDArray
    ls: List[int]
    Pls: List[NDArray]
    ndata: int
    data_vector: NDArray
    data_vector_mask: NDArray

    def __init__(
        self,
        kmin: float,
        kmax: float,
        ls: Union[int, List[int]],
        pkl_path: Location
    ) -> None:
        pkl: NDArray = np.loadtxt(pkl_path, dtype=np.float64)
        self._check_pkl(pkl)
        self._pkl = pkl

        ks = pkl[:, 0]
        kmask = slice(
            np.searchsorted(ks, kmin),
            np.searchsorted(ks, kmax, side='right')
        )
        self.kdata = ks[kmask]

        max_nls = pkl.shape[-1] - 1
        self.ls = self._check_ls(ls, max_nls)
        self.Pls = [pkl[:, 1 + l // 2][kmask] for l in self.ls]

        self.data_vector = np.ravel(np.hstack(self.Pls))
        self.ndata = self.data_vector.size
        bool_kmask = np.zeros(ks.shape[0], dtype=bool)
        bool_kmask[kmask] = True
        all_false = np.zeros(ks.shape[0], dtype=bool)
        data_vector_mask = np.hstack(
            [(bool_kmask if (2 * i in self.ls) else all_false)
             for i in range(max_nls)]
        )
        self.data_vector_mask = data_vector_mask

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
    pkldatas: List[CroppedPklData]
    ndata: int
    data_vector: NDArray
    invcov: NDArray

    def __init__(
        self,
        pklinfo: Union[Dict, List[Dict]],
        cov_path: Location,
        common: Optional[Dict[str, Any]] = None,
        Nreal: Optional[int] = None
    ) -> None:
        if isinstance(pklinfo, dict):
            infolist = [deepcopy(pklinfo)]
        else:
            infolist = deepcopy(pklinfo)
        if common is not None:
            new_infolist = [deepcopy(common) for _ in infolist]
            for raw, new in zip(new_infolist, infolist):
                raw.update(new)
            infolist = new_infolist
        self.pkldatas = [CroppedPklData(**infodict) for infodict in infolist]
        self.ndata = sum([pkldata.ndata for pkldata in self.pkldatas])
        self.data_vector = np.ravel(
            np.hstack([pkldata.data_vector for pkldata in self.pkldatas]))
        assert self.data_vector.shape[0] == self.ndata

        cov = np.loadtxt(cov_path, dtype=np.float64)
        self._check_cov(cov)
        data_vector_mask = np.ravel(
            np.hstack([_.data_vector_mask for _ in self.pkldatas]))
        cov = cov[np.outer(data_vector_mask, data_vector_mask)].reshape(
            (self.ndata, -1))
        self.invcov = np.linalg.inv(cov)
        if Nreal is not None:
            self.invcov *= (Nreal - self.ndata - 2) / (Nreal - 1)
        assert self.invcov.shape[0] == self.ndata

    def _check_cov(self, cov: NDArray) -> None:
        ndim = cov.ndim
        if ndim != 2:
            raise ValueError(f'expect matrix, instead of ndim={ndim}')
        shape = cov.shape
        if shape[0] != shape[-1]:
            raise ValueError(f'expect square matrix, instead of shape={shape}')
        nall = sum([_.data_vector_mask.shape[0] for _ in self.pkldatas])
        if shape[0] != nall:
            raise ValueError('pkl and cov not match')
