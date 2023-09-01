from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from cobaya.log import HasLogger
from scipy.integrate import quad
from scipy.interpolate import interp1d
from .pybird import pybird
from .pybird.pybird import BirdLike
from .transformer import BirdTransformer
from .transformer import PlainBird

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Binning(BirdTransformer, HasLogger):
    """Match the theoretical output to data, do binning and store results

    Parameters
    ----------
    kout: ArrayLike, 1d
        k of data
    accboost: int
        accuracy boost, default 1, do integration using 100 points per bin
    decimals: int
        compute delta_k by rounding the difference of last two kout.
        Default is 2 and this works well when delta_k = 0.01
    co: Common
        this class only uses co.k, default pybird.Common
    name: str
        logger name, by default 'pybird.binning'

    Methods
    -------
    kbinning(bird: Bird): apply binning

    Notes
    -----
    kbins will be constructed using kout[-1] - kout[-2]
    """

    def __init__(
        self,
        kout,
        accboost: int = 1,
        decimals: int = 2,
        co: pybird.Common = pybird.common,
        name: str = "pybird.binning",
        kstart: float | None = None,
        kend: float | None = None,
        nbins: int | None = None,
    ) -> None:
        self.set_logger(name=name)
        self.kout = np.array(kout)
        self.co = co
        self.accboost = accboost
        self.decimals = decimals
        self.mpi_info("binning correction: on")
        self.mpi_info(
            "%d data points, from %.3f to %.3f",
            self.kout.size,
            self.kout[0],
            self.kout[-1],
        )
        if kstart is None and kend is None and nbins is None:
            # XXX: only suitable for dk like: 0.01, 0.005, 0.001, ...
            kspaces = np.around(self.kout[1:] - self.kout[:-1], decimals=decimals)  # type: ignore
            kspace_diff = kspaces[1:] - kspaces[:-1]
            if not np.allclose(kspace_diff, 0, rtol=0, atol=1e-6):
                self.mpi_warning(
                    "binning correction on, "
                    "but given kout seems not linearly spaced, "
                    "be careful because the constructed kbins may be wrong, "
                    "especially when 'kmax' is small",
                )
            self.mpi_info(
                "round the difference of last two kout to %d decimal places",
                self.decimals,
            )
            self.load_binning(self.kout)
            self.mpi_info(
                "inferred kedges=%f, %f, ..., %f",
                self.binmin[0],
                self.binmin[1],
                self.binmax[-1],
            )
        else:
            if kstart is None or kend is None or nbins is None:
                raise ValueError("need specify kstart, kend and nbins together")
            kedges = np.linspace(kstart, kend, nbins + 1)
            ileft = np.searchsorted(kedges, self.kout[0]) - 1
            iright = np.searchsorted(kedges, self.kout[-1], side="right") + 1
            kedges = kedges[ileft:iright]
            self.load_binning(self.kout, kedges=kedges)
            self.mpi_info(
                "kedges: %.3f, %.3f, ..., %.3f", kedges[0], kedges[1], kedges[-1]
            )

        self.mpi_info("num of kgrids in each bin: %d", self.points[0].size)

    def load_binning(self, setkout, kedges=None) -> None:
        """
        Create the bins of the data k's
        """
        if kedges is None:
            delta_k = np.round(setkout[-1] - setkout[-2], self.decimals)
            kcentral = (setkout[-1] - delta_k * np.arange(len(setkout)))[::-1]
            self.binmin = binmin = kcentral - delta_k / 2
            self.binmax = binmax = kcentral + delta_k / 2
        else:
            self.binmin = binmin = kedges[:-1]
            self.binmax = binmax = kedges[1:]
        self.binvol = np.array(
            [
                quad(lambda k: k**2, kbinmin, kbinmax)[0]
                for (kbinmin, kbinmax) in zip(binmin, binmax)
            ]
        )
        self.keff = np.array(
            [
                quad(lambda k: k * k**2, kbinmin, kbinmax)[0]
                for (kbinmin, kbinmax) in zip(binmin, binmax)
            ]
        )
        self.keff = self.keff / self.binvol
        points = [
            np.linspace(kbinmin, kbinmax, 100 * self.accboost)
            for (kbinmin, kbinmax) in zip(binmin, binmax)
        ]
        self.points = np.array(points)

    def integrBinning(self, P: NDArray) -> NDArray:
        """
        Integrate over each bin of the data k's
        """
        Pkint = interp1d(
            self.co.k,
            P,
            axis=-1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )
        res = np.trapz(Pkint(self.points) * self.points**2, x=self.points, axis=-1)
        return res / self.binvol

    def kbinning(self, bird: BirdLike):
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        return PlainBird(
            f=bird.f,
            co=bird.co,
            P11l=self.integrBinning(bird.P11l),
            Ploopl=self.integrBinning(bird.Ploopl),
            Pctl=self.integrBinning(bird.Pctl),
            PctNNLOl=self.integrBinning(bird.PctNNLOl),
            Pstl=self.integrBinning(bird.Pstl),
            Picc=self.integrBinning(bird.Picc),
        )

    def transform(self, birdlike: BirdLike):
        return self.kbinning(birdlike)
