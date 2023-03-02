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

    def _effective_z(
        x1, y1, z1, redshift1, weight1, x2, y2, z2, redshift2, weight2, smin, smax
    ):
        ndata1 = x1.size
        ndata2 = x2.size
        numerator = 0.0
        denominator = 0.0
        smin2 = smin**2
        smax2 = smax**2
        for i in tqdm(range(ndata1)):
            for j in range(ndata2):
                distance = (
                    (x1[i] - x2[j]) ** 2 + (y1[i] - y2[j]) ** 2 + (z1[i] - z2[j]) ** 2
                )
                if smin2 <= distance <= smax2:
                    numerator += (
                        weight1[i] * weight2[j] * (redshift1[i] + redshift2[j]) / 2
                    )
                    denominator += weight1[i] * weight2[j]
        return numerator / denominator

else:

    @njit(parallel=True)
    def _effective_z(
        x1, y1, z1, redshift1, weight1, x2, y2, z2, redshift2, weight2, smin, smax
    ):
        ndata1 = x1.size
        ndata2 = x2.size
        numerator = 0.0
        denominator = 0.0
        smin2 = smin**2
        smax2 = smax**2
        for i in prange(ndata1):
            for j in range(ndata2):
                distance = (
                    (x1[i] - x2[j]) ** 2 + (y1[i] - y2[j]) ** 2 + (z1[i] - z2[j]) ** 2
                )
                if smin2 <= distance <= smax2:
                    numerator += (
                        weight1[i] * weight2[j] * (redshift1[i] + redshift2[j]) / 2
                    )
                    denominator += weight1[i] * weight2[j]
        return numerator / denominator


def effective_z(
    RA1,
    DEC1,
    redshift1,
    weight1,
    RA2=None,
    DEC2=None,
    redshift2=None,
    weight2=None,
    omegam: float = 0.307115,
    smin: float = 25,
    smax: float = 150,
):
    dist1 = comoving_distance(omegam)(redshift1)
    x1, y1, z1 = sky_to_castesian(RA1, DEC1, dist1)
    if any(x is None for x in [RA2, DEC2, redshift2, weight2]):
        x2, y2, z2, redshift2, weight2 = x1, y1, z1, redshift1, weight1
    else:
        dist2 = comoving_distance(omegam)(redshift2)
        x2, y2, z2 = sky_to_castesian(RA2, DEC2, dist2)
    return _effective_z(
        x1, y1, z1, redshift1, weight1, x2, y2, z2, redshift2, weight2, smin, smax
    )
