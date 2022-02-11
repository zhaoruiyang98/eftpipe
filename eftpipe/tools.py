import os
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import (
    Any,
    List,
    TYPE_CHECKING,
    Iterable,
    Dict,
    Optional,
    Union,
    Tuple,
    Set,
    cast,
)
from eftpipe.typing import SimpleYaml

# This Mixin is not reliable, don't use it
class SampledParamsMixin:

    @property
    def nsampled(self) -> int:
        return len(self.sampled_params)

    @property
    def sampled_params(self) -> List[str]:
        input_params = set(self.input_params)
        sampled_params = set(
            self.provider.model.parameterization.sampled_params().keys()
        )
        return list(input_params.intersection(sampled_params))

    if TYPE_CHECKING:
        def __getattr__(self, __name: str) -> Any: ...


def update_path_in_dict(d: SimpleYaml, base: Path) -> None:
    if isinstance(d, dict):
        for key, value in d.items():
            if not isinstance(value, (dict, list)):
                if 'path' in key:
                    d[key] = str(base / str(d[key]))
            else:
                update_path_in_dict(value, base)
    if isinstance(d, list):
        for item in d:
            update_path_in_dict(item, base)


class PathContext:
    def __init__(self, path) -> None:
        self.old = Path('.').resolve()
        self.new = self.old / path

    def __enter__(self):
        os.chdir(self.new)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.old)


class RandomParams:
    """Generate Random Params Dictionary given reference mean and covmat

    Parameters
    ----------
    ref: dict[str, float] | dict[str, dict[str, float]]
        parameter names and the corresponding mean values
        if ref is nestes dictionary, the level 0 keys are interpreted as group names
    covmat: ArrayLike
        covariance matrix of sampled parameters
    covmat_params: Iterable[str]
        parameter names matches covmat
    seed: int
        random seed for rng, default None

    Attributes
    ----------
    sampled_names: list[str]
        names of sampled parameters
    covmat: ndarray, 2d, float64
        covariance matrix of sampled params
    mean: ndarray, 1d, float64
        the mean value of sampled params
    rng: Generator
        random number generator
    count: int
        called number of `random`

    Methods
    -------
    fix(names=None, groups=None)
        fix some parameters or parameters group
    random()
        generate random params dict, the first point is always the mean
    """

    def __init__(
        self,
        ref: Union[Dict[str, float], Dict[str, Dict[str, float]]],
        covmat: npt.ArrayLike,
        covmat_params: Iterable[str],
        seed: Optional[int] = None,
    ) -> None:
        validref, groups = self._unpack_grouped_ref(ref)
        self._groups = groups
        self._check(validref, covmat, covmat_params)
        self.covmat = np.asarray(covmat)
        self._params_dict = validref.copy()
        self.sampled_names = [k for k in covmat_params]
        self.mean = np.array([self._params_dict[k]
                             for k in self.sampled_names])
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def random(self) -> Dict[str, float]:
        if self.count == 0:
            res = self._params_dict.copy()
        else:
            res = self._params_dict.copy()
            sampled_values = \
                np.random.multivariate_normal(self.mean, self.covmat)
            for name, value in zip(self.sampled_names, sampled_values):
                res[name] = value
        self.count += 1
        return res

    def fix(self, names=None, groups=None):
        if not (names or groups):
            return
        if names is None:
            names = {}
        if groups is None:
            groups = {}
        fixed = set()
        for x in names:
            fixed.add(x)
        for name in groups:
            fixed = fixed.union(self._groups[name])
        new_sampled_names = [x for x in self.sampled_names if x not in fixed]
        need_delete = [x for x in self.sampled_names if x in fixed]
        inds = [self.sampled_names.index(x) for x in need_delete]
        self.covmat = np.delete(self.covmat, inds, axis=0)
        self.covmat = np.delete(self.covmat, inds, axis=1)
        self.sampled_names = new_sampled_names
        self._params_dict = {
            k: self._params_dict[k] for k in new_sampled_names}
        self.mean = np.array(
            [self._params_dict[k] for k in self.sampled_names])

    def _unpack_grouped_ref(
        self,
        ref: Union[Dict[str, float], Dict[str, Dict[str, float]]]
    ) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
        grouped = False
        for _, v in ref.items():
            if isinstance(v, dict):
                grouped = True
                break
        if grouped:
            ref = cast(Dict[str, Dict[str, float]], ref)
            outgroup = {group: set(dct.keys()) for group, dct in ref.items()}
            outref = {}
            for _, dct in ref.items():
                for name, value in dct.items():
                    outref[name] = value
            out = outref, outgroup
        else:
            ref = cast(Dict[str, float], ref)
            out = ref, {"__all__": set(ref.keys())}
        return out

    def _check(
        self,
        ref: Dict[str, Any],
        covmat: npt.ArrayLike,
        covmat_params: Iterable[str],
    ) -> None:
        params_names = ref.keys()
        cover_all_params = set(covmat_params).issubset(set(params_names))
        if not cover_all_params:
            raise ValueError(
                "some parameters in covmat_params are not covered by ref")
        covmat = np.asarray(covmat)
        if len(covmat) != len(covmat):
            raise ValueError(
                "length of covmat_params doesn't match covmat")
        if covmat.ndim != 2:
            raise TypeError(f"expect 2d matrix")
        if covmat.shape[0] != covmat.shape[1]:
            raise TypeError(f"expect square matrix")
