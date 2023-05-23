from __future__ import annotations
import getdist
import numpy as np
from typing import Iterable, TypedDict, TYPE_CHECKING
from cobaya import Likelihood
from scipy.interpolate import (
    interp1d,
    RBFInterpolator,
    RectBivariateSpline,
    RegularGridInterpolator,
)

if TYPE_CHECKING:
    from typing_extensions import NotRequired


def str_or_list(x: str | Iterable[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def pivot_table(
    data,
    value_index: int = -1,
    index: list[int] | None = None,
    rowvar: bool = False,
    fill_value: float = 0.0,
):
    """convert dataframe-like data to values on mesh

    Parameters
    ----------
    data : array_like
        data to convert
    value_index : int, optional
        index of the column to use as values, by default -1
    index : list[int], optional
        list of indices of the columns to use as index, by default all but the last
    rowvar : bool, optional
        If rowvar is false (default), then each column represents a variable,
        with observations in the row. Otherwise, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
    fill_value : float, optional
        value to replace missing values with, by default 0.0
    """
    data = np.asarray(data)
    if not rowvar:
        data = data.T
    if not index:
        all_indices = set(range(data.shape[0]))
        all_indices.remove(
            value_index if value_index >= 0 else data.shape[0] + value_index
        )
        index = list(all_indices)
    coords, pos = [], []
    for i in index:
        x, y = np.unique(data[i], return_inverse=True)
        coords.append(x)
        pos.append(y)
    pivot_table = np.empty(tuple([x.size for x in coords]), dtype=data.dtype)
    pivot_table.fill(fill_value)
    pivot_table[tuple(pos)] = data[value_index]
    return tuple(coords), pivot_table


class SamplesInfo(TypedDict):
    chain_root: str
    params: list[str] | str
    girdlike: NotRequired[bool]
    settings: NotRequired[dict]


def interpolate_sample(sample: SamplesInfo):
    chains = getdist.loadMCSamples(
        sample["chain_root"], settings=sample.get("settings", {})
    )
    params = str_or_list(sample["params"])
    # chains.loglikes actually gives -logpost
    logpost = -chains.loglikes  # type: ignore
    nparams = len(params)

    if nparams == 0:
        raise ValueError("Must specify at least one parameter")
    if nparams == 1:
        return interp1d(
            chains[params[0]],
            logpost,
            kind="cubic",
            bounds_error=False,
            fill_value=-np.inf,
        )
    if not sample.get("gridlike", False):
        return RBFInterpolator(np.array([chains[p] for p in params]).T, logpost)
    coords, z = pivot_table(
        np.vstack([chains[p] for p in params] + [logpost]).T, fill_value=-np.inf  # type: ignore
    )
    if nparams == 2:
        x, y = coords
        return RectBivariateSpline(x, y, z, kx=3, ky=3)
    return RegularGridInterpolator(coords, z, bounds_error=False, fill_value=-np.inf)


class GridLike(Likelihood):
    """used to combined existing independent chains."""

    samples: list[SamplesInfo]

    def initialize(self):
        if not isinstance(self.samples, list):
            self.samples = [self.samples]
        interpolator = []
        params_per_like: list[list[str]] = []
        for sample in self.samples:
            params_per_like.append(str_or_list(sample["params"]))
            interpolator.append(interpolate_sample(sample))

        self.params_per_like = params_per_like
        self.interpolator = interpolator
        self.requires = list(set([item for ps in params_per_like for item in ps]))

    def logp(self, **params_values):
        retval = sum(
            fn(np.array([params_values[p] for p in params])[None, :])
            for fn, params in zip(self.interpolator, self.params_per_like)
        )
        return retval
