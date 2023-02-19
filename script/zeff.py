from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from scipy.integrate import quad


def sky_to_castesian(RA, DEC, dist, is_radian: bool = False):
    if not is_radian:
        RA = RA * np.pi / 180
        DEC = DEC * np.pi / 180
    x = dist * np.cos(DEC) * np.cos(RA)
    y = dist * np.cos(DEC) * np.sin(RA)
    z = dist * np.sin(DEC)
    return x, y, z


@dataclass
class comoving_distance:
    omegam: float
    hubble: Callable[[Any], Any] = field(init=False)

    def __post_init__(self):
        self.hubble = lambda z: 100 * np.sqrt(
            self.omegam * (1.0 + z) ** 3 + 1 - self.omegam
        )
        self.kernel = lambda z: 299792.458 / self.hubble(z)

    def __call__(
        self, z, epsabs: float = 1.49e-8, epsrel: float = 1.49e-8, limit: int = 50
    ):
        z = np.atleast_1d(z)
        retval = np.empty_like(z)
        for i, zi in enumerate(z.flat):
            retval.flat[i] = quad(
                self.kernel, 0, zi, epsabs=epsabs, epsrel=epsrel, limit=limit
            )[0]
        if z.size == 1:
            return retval[0]
        return retval


try:
    from numba import njit, prange
except ImportError:

    from tqdm import tqdm

    def _effective_z(x, y, z, redshift, weight, smin, smax):
        ndata = x.size
        numerator = 0.0
        denominator = 0.0
        smin2 = smin**2
        smax2 = smax**2
        for i in tqdm(range(ndata)):
            for j in range(ndata):
                if i == j:
                    continue
                distance = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
                if smin2 <= distance <= smax2:
                    numerator += weight[i] * weight[j] * (redshift[i] + redshift[j]) / 2
                    denominator += weight[i] * weight[j]
        return numerator / denominator

else:

    @njit(["f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8, f8)"], parallel=True)
    def _effective_z(x, y, z, redshift, weight, smin, smax):
        ndata = x.size
        numerator = 0.0
        denominator = 0.0
        smin2 = smin**2
        smax2 = smax**2
        for i in prange(ndata):
            for j in range(ndata):
                if i == j:
                    continue
                distance = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
                if smin2 <= distance <= smax2:
                    numerator += weight[i] * weight[j] * (redshift[i] + redshift[j]) / 2
                    denominator += weight[i] * weight[j]
        return numerator / denominator


def effective_z(
    RA,
    DEC,
    redshift,
    weight,
    omegam: float = 0.307,
    smin: float = 25,
    smax: float = 150,
):
    dist = comoving_distance(omegam)(redshift)
    x, y, z = sky_to_castesian(RA, DEC, dist)
    return _effective_z(x, y, z, redshift, weight, smin, smax)
