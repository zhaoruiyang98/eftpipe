from __future__ import annotations
import inspect
import itertools
import logging
import os
import sys
import time
import numpy as np
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    cast,
    Callable,
    Container,
    Dict,
    Generic,
    Iterable,
    TypeVar,
    TYPE_CHECKING,
)
from cobaya.log import LoggedError
from cobaya.mpi import is_main_process
from cobaya.mpi import root_only  # DO NOT delete this line
from .etyping import SimpleYaml
from .etyping import SupportsRichComparisonT

if TYPE_CHECKING:
    import numpy.typing as npt
    from logging import Logger

_T = TypeVar("_T")


# This Mixin is not reliable, do not use it
class SampledParamsMixin:
    @property
    def nsampled(self) -> int:
        return len(self.sampled_params)

    @property
    def sampled_params(self) -> list[str]:
        input_params = set(self.input_params)
        sampled_params = set(
            self.provider.model.parameterization.sampled_params().keys()
        )
        return list(input_params.intersection(sampled_params))

    if TYPE_CHECKING:

        def __getattr__(self, __name: str) -> Any:
            ...


def update_path_in_dict(
    d: SimpleYaml,
    base: Path,
    extra: Container[str] = (),
) -> None:
    if isinstance(d, dict):
        for key, value in d.items():
            if not isinstance(value, (dict, list)):
                if "path" in key:
                    d[key] = str(base / str(d[key]))
                elif key in extra:
                    d[key] = str(base / str(d[key]))
            else:
                update_path_in_dict(value, base, extra=extra)
    if isinstance(d, list):
        for item in d:
            update_path_in_dict(item, base, extra=extra)


class _NotFound(Enum):
    NOTFOUND = 1


NOTFOUND = _NotFound.NOTFOUND


def recursively_update_dict(default: dict, override: dict):
    for k, v in override.items():
        rawv = default.get(k, NOTFOUND)
        if rawv is NOTFOUND:
            default[k] = v
        else:
            if isinstance(rawv, dict):
                recursively_update_dict(rawv, v)
            else:
                default[k] = v


def set_value_in_nested_dict(dct: dict[str, Any], value, *keys: str):
    *init, last = keys
    anchor = dct
    for key in init:
        # not found or None or dict
        newanchor = anchor.get(key, None)
        # if value is None, or missing key, create an empty dict.
        if newanchor is None:
            newanchor = {}
            anchor[key] = newanchor
        anchor = newanchor
    anchor[last] = value


def bool_or_list(x: bool | Iterable[bool]) -> list[bool]:
    return [x] if isinstance(x, bool) else list(x)


def int_or_list(x: int | Iterable[int]) -> list[int]:
    return [x] if isinstance(x, int) else list(x)


def str_or_list(x: str | Iterable[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def group_lists(*args: list[SupportsRichComparisonT]) -> list[SupportsRichComparisonT]:
    """sort and join lists"""
    out: set[SupportsRichComparisonT] = set()
    return sorted(out.union(*args))


def replace_suffix(path: Path, suffix: str) -> Path:
    """support invalid suffix"""
    return path.parent / (path.stem + suffix)


if sys.version_info >= (3, 10):
    pairwise = itertools.pairwise
else:

    def pairwise(x: Iterable[_T]) -> zip[tuple[_T, _T]]:
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = itertools.tee(x)
        next(b, None)
        return zip(a, b)


@contextmanager
def timer(name: str = "", info: Callable = print):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    if is_main_process():
        info(f"{name} took {end - start} seconds")


@contextmanager
def do_nothing(*args, **kwargs):
    yield


@contextmanager
def disable_logging():
    logging.disable()
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


@contextmanager
def verbose_guard(verbose: bool):
    with do_nothing() if verbose else disable_logging():
        yield


class Initializer(Generic[_T]):
    """delayed initialization"""

    def __init__(
        self, cls: Callable[..., _T], dct: dict[str, Any], logger: str | Logger
    ) -> None:
        args: list[str] = []
        kwargs: dict[str, Any] = {}
        for k, v in inspect.signature(cls).parameters.items():
            if v.kind is v.VAR_KEYWORD:
                continue
            if v.kind is v.VAR_POSITIONAL:
                raise LoggedError(
                    logger, "%s does not support *args", self.__class__.__name__
                )
            default = v.default
            if default is v.empty:
                args.append(k)
            else:
                kwargs[k] = default
        for k, v in dct.items():
            if k in args:
                args.remove(k)
            elif k not in kwargs:
                raise LoggedError(logger, "%r does not have keyword %s", cls, k)
            kwargs[k] = v
        self.cls = cls
        self.kwargs = kwargs
        for k in args:
            if k not in self.kwargs:
                raise LoggedError(logger, "missing positional argument %s", k)

    def initialize(self, **override) -> _T:
        return self.cls(**{**self.kwargs, **override})


@contextmanager
def PathContext(path: str | os.PathLike):
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


def banner(msg: str, width: int, fill_symbol: str = "="):
    mgs_len = len(msg)
    if mgs_len > width:
        raise ValueError("cannot fit message in banner")
    nleft = (width - mgs_len) // 2
    nright = width - mgs_len - nleft
    return fill_symbol * nleft + msg + fill_symbol * nright


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
        called number of ``random``

    Methods
    -------
    fix(names=None, groups=None)
        fix some parameters or parameters group
    random()
        generate random params dict, the first point is always the mean
    """

    def __init__(
        self,
        ref: dict[str, float] | dict[str, dict[str, float]],
        covmat: npt.ArrayLike,
        covmat_params: Iterable[str],
        seed: int | None = None,
    ) -> None:
        validref, groups = self._unpack_grouped_ref(ref)
        self._groups = groups
        self._check(validref, covmat, covmat_params)
        self.covmat = np.asarray(covmat)
        self._params_dict = validref.copy()
        self.sampled_names = [k for k in covmat_params]
        self.mean = np.array([self._params_dict[k] for k in self.sampled_names])
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def random(self) -> dict[str, float]:
        if self.count == 0:
            res = self._params_dict.copy()
        else:
            res = self._params_dict.copy()
            sampled_values = np.random.multivariate_normal(self.mean, self.covmat)
            for name, value in zip(self.sampled_names, sampled_values):
                res[name] = value
        self.count += 1
        return res

    def fix(
        self, names: Iterable[str] | None = None, groups: Iterable[str] | None = None
    ):
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
        if not new_sampled_names:
            raise ValueError("All parameters are fixed!")
        need_delete = [x for x in self.sampled_names if x in fixed]
        inds = [self.sampled_names.index(x) for x in need_delete]
        self.covmat = np.delete(self.covmat, inds, axis=0)
        self.covmat = np.delete(self.covmat, inds, axis=1)
        self.sampled_names = new_sampled_names
        self.mean = np.array([self._params_dict[k] for k in self.sampled_names])

    def _unpack_grouped_ref(
        self, ref: dict[str, float] | dict[str, dict[str, float]]
    ) -> tuple[dict[str, float], dict[str, set[str]]]:
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
        ref: dict[str, Any],
        covmat: npt.ArrayLike,
        covmat_params: Iterable[str],
    ) -> None:
        params_names = ref.keys()
        cover_all_params = set(covmat_params).issubset(set(params_names))
        if not cover_all_params:
            raise ValueError("some parameters in covmat_params are not covered by ref")
        covmat = np.asarray(covmat)
        if len(covmat) != len(covmat):
            raise ValueError("length of covmat_params doesn't match covmat")
        if covmat.ndim != 2:
            raise TypeError(f"expect 2d matrix")
        if covmat.shape[0] != covmat.shape[1]:
            raise TypeError(f"expect square matrix")
